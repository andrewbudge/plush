use clap::{Parser, Subcommand};
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

mod index;
mod progress;
mod sketch;

use index::{HeaderFormat, load_mmap_index, parse_header};
use progress::{HumanDuration, Progress};

// One output hit: (hit_id, hit_accession, bin_distance, similarity, protein_len, protein)
type RawHit = (String, String, u32, u32, usize, String);
// Per-query result: (query_id, hits, (cand_count, verify_count, too_short))
type QResult = (String, Vec<RawHit>, (usize, usize, bool));

fn sketch_query(
    protein: &[u8],
    views: &index::IndexViews,
    min_sim: u32,
) -> (Option<RawHit>, usize, usize) {
    let qsk = sketch::sketch(protein);
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for k in sketch::band_keys(&qsk) {
        for idx in views.bands().find(k) {
            seen.insert(views.bands().ref_at(idx));
        }
    }
    let cand_count = seen.len();
    let mut best: Option<(u32, u32)> = None;
    for &rid in &seen {
        let Some(rsk) = views.sketch(rid as usize) else {
            continue;
        };
        let m = sketch::matching_bins(&qsk, rsk);
        if best.is_none_or(|b| m > b.1 || (m == b.1 && rid < b.0)) {
            best = Some((rid, m));
        }
    }
    let Some((rid, m)) = best else {
        return (None, cand_count, cand_count);
    };
    let similarity = sketch::estimate_pct(m);
    if similarity < min_sim {
        return (None, cand_count, cand_count);
    }
    let Some((bid, accession, _organism)) = views.cold_meta(rid as usize) else {
        return (None, cand_count, cand_count);
    };
    let protein_str = std::str::from_utf8(protein).unwrap().to_string();
    (
        Some((
            bid,
            accession,
            sketch::BINS as u32 - m,
            similarity,
            protein.len(),
            protein_str,
        )),
        cand_count,
        cand_count,
    )
}

fn file_bytes(path: &PathBuf) -> usize {
    std::fs::metadata(path).map(|m| m.len() as usize).unwrap_or(0)
}

// Cheap pass to get an exact query count for the search progress bar's goal.
fn count_records(path: &PathBuf) -> usize {
    let mut reader = needletail::parse_fastx_file(path).expect("open query");
    let mut n = 0usize;
    while let Some(record) = reader.next() {
        record.expect("record");
        n += 1;
    }
    n
}

#[derive(Parser)]
#[command(name = "plush", version, about = "Alignment-free protein search (binned MinHash sketch)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Index a reference protein FASTA into a .plsh file
    Index {
        /// Input protein FASTA (default: first whitespace token of the header is the id)
        #[arg(short, long)]
        input: PathBuf,
        /// Header ID format: `token` (first whitespace token) or `pipe` (`id|seqid ... [organism]`)
        #[arg(long, value_enum, default_value = "token")]
        header_format: HeaderFormat,
        /// Output .plsh index path
        #[arg(short = 'x', long)]
        output: PathBuf,
        /// Minimum protein length in AA (absolute floor 6)
        #[arg(short = 'm', long, default_value_t = 50)]
        min_len: usize,
        /// Limit number of sequences processed (for testing)
        #[arg(long)]
        limit: Option<usize>,
        /// Number of temp hash buckets to stream through before aggregating
        /// into one .plsh. Set when RAM is limited (0 = no sharding).
        #[arg(long, default_value_t = 0)]
        shards: usize,
    },
    /// Search query proteins against a .plsh index
    Search {
        /// Query protein FASTA/FASTQ (default: first whitespace token of the header is the id)
        #[arg(short, long)]
        input: PathBuf,
        /// Header ID format: `token` (first whitespace token) or `pipe` (text before the first `|`)
        #[arg(long, value_enum, default_value = "token")]
        header_format: HeaderFormat,
        /// Path to the .plsh index
        #[arg(short = 'x', long)]
        index: PathBuf,
        /// Minimum estimated similarity (0-100) to report a hit
        #[arg(short = 's', long, default_value_t = 95)]
        min_sim: u32,
        /// Minimum protein length in AA for search
        #[arg(short = 'm', long, default_value_t = 50)]
        min_len: usize,
        /// Print performance statistics (candidate sets, verify calls, throughput) to stderr
        #[arg(long)]
        perf_stats: bool,
        /// Output file for hits TSV (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output matching proteins to FASTA file (optional)
        #[arg(short = 'f', long)]
        output_proteins: Option<PathBuf>,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Index {
            input,
            output,
            min_len,
            limit,
            shards,
            header_format,
        } => {
            let t0 = Instant::now();
            let bytes_total = file_bytes(&input);
            if shards > 1 {
                let shard_dir = output.with_extension("shardtmp");
                eprintln!("sketching {} (sharded x{})...", input.display(), shards);
                let mut prog = Progress::start(
                    bytes_total,
                    "{bar:40} {bytes}/{total_bytes} sketched (eta {eta})",
                );
                let (shards_list, stats) = index::collect_shards(
                    &input,
                    min_len,
                    limit,
                    header_format,
                    &shard_dir,
                    shards,
                    &mut |bytes| prog.set(bytes),
                )
                .expect("collect shards");
                prog.finish();
                drop(prog);

                let total = stats.kept as u64;
                eprintln!("aggregating {} shards into {}...", shards, output.display());
                let mut prog = Progress::start(
                    total as usize,
                    "{bar:40} {pos}/{len} refs aggregated (eta {eta})",
                );
                index::aggregate_shards(&output, &shards_list, total, &mut |n| prog.set(n))
                    .expect("write index");
                prog.finish();
                drop(prog);
                let _ = std::fs::remove_dir_all(&shard_dir);
                eprintln!(
                    "{} refs indexed ({} short) in {}",
                    stats.kept,
                    stats.skipped_short,
                    HumanDuration(t0.elapsed())
                );
            } else {
                eprintln!("sketching {}...", input.display());
                let mut prog = Progress::start(
                    bytes_total,
                    "{bar:40} {bytes}/{total_bytes} sketched (eta {eta})",
                );
                let (hits, stats) = index::collect_hits(
                    &input,
                    min_len,
                    limit,
                    header_format,
                    &mut |bytes| prog.set(bytes),
                );
                prog.finish();
                drop(prog);

                eprintln!("writing {}...", output.display());
                let mut prog = Progress::start(
                    hits.len(),
                    "{bar:40} {pos}/{len} refs written (eta {eta})",
                );
                index::write_index(&output, &hits, &mut |n| prog.set(n)).expect("write index");
                prog.finish();
                drop(prog);
                eprintln!(
                    "{} refs indexed ({} short) in {}",
                    stats.kept,
                    stats.skipped_short,
                    HumanDuration(t0.elapsed())
                );
            }
        }
        Commands::Search {
            input,
            header_format,
            index,
            min_sim,
            min_len,
            perf_stats,
            output,
            output_proteins,
        } => {
            let t0 = Instant::now();
            eprintln!("loading index {}...", index.display());
            let mmap_index = load_mmap_index(&index).expect("load index");
            let n_refs = mmap_index.n_hits();

            let views = mmap_index.views();
            debug_assert_eq!(views.n_hits(), n_refs);

            let mut out: Box<dyn std::io::Write> = match &output {
                Some(p) => Box::new(std::io::BufWriter::new(std::fs::File::create(p).unwrap())),
                None => Box::new(std::io::stdout().lock()),
            };
            writeln!(
                out,
                "status\tquery_id\thit_id\thit_accession\tbin_distance\tsimilarity\tprotein_len"
            )
            .unwrap();

            let mut protein_out: Option<Box<dyn std::io::Write>> = output_proteins.as_ref().map(|p| {
                Box::new(std::io::BufWriter::new(std::fs::File::create(p).unwrap())) as Box<dyn std::io::Write>
            });

            // Draws to stderr and auto-hides on a non-TTY/piped stderr, so this
            // never touches the TSV on stdout; progress falls back to plain
            // stderr lines when hidden (e.g. logs under nohup/slurm). The
            // extra pass to count queries is cheap relative to sketching/search
            // and gives an exact "X/X queries searched" bar.
            let n_total = count_records(&input);
            let mut prog = Progress::start(
                n_total,
                "{bar:40} {pos}/{len} queries searched (eta {eta})",
            );

            let mut reader = needletail::parse_fastx_file(&input).expect("open query");
            let mut n_queries = 0usize;
            let mut n_hits = 0usize;
            let mut n_too_short = 0usize;
            let mut n_no_candidate = 0usize;
            let mut cand_sum: usize = 0;
            let mut verify_sum: usize = 0;
            let mut max_cand: usize = 0;
            const BATCH: usize = 4096;
            let mut batch: Vec<(String, Vec<u8>)> = Vec::with_capacity(BATCH);
            loop {
                batch.clear();
                while batch.len() < BATCH {
                    match reader.next() {
                        Some(Ok(seqrec)) => {
                            let raw = String::from_utf8_lossy(seqrec.id());
                            let (qid, _, _) = parse_header(&raw, header_format);
                            batch.push((qid, seqrec.seq().to_vec()));
                        }
                        Some(Err(e)) => panic!("read error: {e}"),
                        None => break,
                    }
                }
                if batch.is_empty() {
                    break;
                }
                n_queries += batch.len();
                let results: Vec<QResult> = batch
                    .par_iter()
                    .map(|(id, seq)| {
                        let protein = seq;
                        if protein.len() < min_len.max(index::MIN_LEN_FLOOR) {
                            return (id.clone(), Vec::new(), (0, 0, true));
                        }
                        let (hit, cand_count, verify_count) = sketch_query(protein, &views, min_sim);
                        match hit {
                            Some(h) => (id.clone(), vec![h], (cand_count, verify_count, false)),
                            None => (id.clone(), Vec::new(), (cand_count, verify_count, false)),
                        }
                    })
                    .collect();
                for (id, hits_this_query, stats) in results {
                    let (cc, vv, ts) = stats;
                    cand_sum += cc;
                    verify_sum += vv;
                    if ts { n_too_short += 1; }
                    if cc == 0 && !ts { n_no_candidate += 1; }
                    if cc > max_cand { max_cand = cc; }
                    if hits_this_query.is_empty() {
                        writeln!(out, "U\t{}\t\t\t0\t0\t0", id).unwrap();
                    } else {
                        n_hits += hits_this_query.len();
                        for (bid, accession, bin_distance, similarity, protein_len, protein_str) in hits_this_query {
                            writeln!(
                                out,
                                "C\t{}\t{}\t{}\t{}\t{}\t{}",
                                id, bid, accession, bin_distance, similarity, protein_len
                            ).unwrap();
                            if let Some(ref mut protein_out) = protein_out {
                                use std::io::Write;
                                let header = format!(">{} hit={} sim={} len={}", id, bid, similarity, protein_len);
                                writeln!(protein_out, "{}\n{}", header, protein_str).unwrap();
                            }
                        }
                    }
                }
                if prog.is_hidden() {
                    eprintln!(
                        "processed {}/{} queries, {} hits, {}",
                        n_queries,
                        n_total,
                        n_hits,
                        HumanDuration(t0.elapsed())
                    );
                } else {
                    prog.set(n_queries);
                }
            }
            prog.finish();
            drop(prog);
            let mut summary = format!("{} hits found in {}", n_hits, HumanDuration(t0.elapsed()));
            if let Some(p) = &output {
                summary.push_str(&format!(", written to {}", p.display()));
            }
            if let Some(p) = &output_proteins {
                summary.push_str(&format!(" (proteins written to {})", p.display()));
            }
            eprintln!("{summary}");
            if perf_stats {
                let secs = t0.elapsed().as_secs_f64();
                eprintln!(
                    "perf: queries={} hits={} too_short={} no_candidate={} cand_sum={} verify_calls={} mean_cand={:.2} max_cand={} queries/s={:.0}",
                    n_queries,
                    n_hits,
                    n_too_short,
                    n_no_candidate,
                    cand_sum,
                    verify_sum,
                    if n_queries > 0 { cand_sum as f64 / n_queries as f64 } else { 0.0 },
                    max_cand,
                    if secs > 0.0 { n_queries as f64 / secs } else { 0.0 },
                );
            }
            drop(out);
        }
    }
}

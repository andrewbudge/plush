use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"PLSH";
const VERSION: u32 = 10;

/// Hard minimum protein length in AA. Sequences shorter than this produce no
/// 6-mer windows and degenerate to an all-ones sketch that would match any
/// other such sketch.
pub const MIN_LEN_FLOOR: usize = 6;

/// Sketch size on disk: 128 bins x 4 bits.
const SKETCH_BIN: usize = crate::sketch::BYTES;
/// Fixed-width hot record: sketch + u32 LE protein length.
const HOT_STRIDE: usize = SKETCH_BIN + 4;
/// MAGIC(4) + VERSION(4) + n_hits(8) + hot_stride(4).
const HEADER_LEN: usize = 20;

#[derive(Clone)]
pub struct RefHit {
    pub id: String,
    pub accession: Box<str>,
    pub organism: String,
    pub length: u32,
    pub sketch: Box<[u8]>,
}

impl serde::Serialize for RefHit {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut st = s.serialize_struct("RefHit", 5)?;
        use serde::ser::SerializeStruct;

        st.serialize_field("sketch", &self.sketch[..])?;
        st.serialize_field("id", &self.id)?;
        st.serialize_field("accession", &self.accession)?;
        st.serialize_field("organism", &self.organism)?;
        st.serialize_field("length", &self.length)?;
        st.end()
    }
}

impl<'de> serde::Deserialize<'de> for RefHit {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Raw {
            sketch: Vec<u8>,
            id: String,
            accession: Box<str>,
            organism: String,
            length: u32,
        }
        let raw = Raw::deserialize(d)?;
        Ok(RefHit {
            id: raw.id,
            accession: raw.accession,
            organism: raw.organism,
            length: raw.length,
            sketch: raw.sketch.into_boxed_slice(),
        })
    }
}

/// Tally of what a full pass over a reference FASTA produced.
pub struct IndexStats {
    pub kept: usize,
    pub skipped_short: usize,
}

pub fn collect_hits(
    input: &Path,
    min_len: usize,
    limit: Option<usize>,
    format: HeaderFormat,
    progress: &mut dyn FnMut(usize),
) -> (Vec<RefHit>, IndexStats) {
    let mut ball = Vec::new();
    let stats =
        collect_hits_sink(input, min_len, limit, format, progress, &mut |h| ball.push(h.clone()));
    (ball, stats)
}

// Stream sketched refs so indexing can shard without holding all hits in RAM.
// `progress` receives approximate bytes consumed from `input` at chunk boundaries.
pub fn collect_hits_sink(
    input: &Path,
    min_len: usize,
    limit: Option<usize>,
    format: HeaderFormat,
    progress: &mut dyn FnMut(usize),
    sink: &mut dyn FnMut(&RefHit),
) -> IndexStats {
    let mut reader = needletail::parse_fastx_file(input).expect("open fasta");
    let mut kept = 0usize;
    let mut skipped_short = 0usize;
    let mut chunk: Vec<(String, Vec<u8>)> = Vec::with_capacity(4096);

    let flush = |chunk: &mut Vec<(String, Vec<u8>)>,
                     kept: &mut usize,
                     sink: &mut dyn FnMut(&RefHit)| {
        use rayon::prelude::*;
        let made: Vec<RefHit> = chunk
            .par_iter()
            .map(|(id, seq)| {
                let (ref_id, accession, organism) = parse_header(id, format);
                RefHit {
                    id: ref_id,
                    accession: accession.into_boxed_str(),
                    organism,
                    length: seq.len() as u32,
                    sketch: crate::sketch::sketch(seq).to_vec().into_boxed_slice(),
                }
            })
            .collect();
        for h in made {
            sink(&h);
            *kept += 1;
            if let Some(lim) = limit && *kept >= lim {
                return;
            }
        }
        chunk.clear();
    };

    let mut done = false;
    let mut bytes_done = 0usize;
    while let Some(record) = reader.next() {
        let seqrec = record.expect("record");
        let seq = seqrec.seq();
        let length = seq.len();
        bytes_done += seqrec.id().len() + length + seqrec.qual().map_or(0, |q| q.len()) + 2;
        if length < min_len.max(MIN_LEN_FLOOR) {
            skipped_short += 1;
            continue;
        }
        let id = String::from_utf8_lossy(seqrec.id()).into_owned();
        chunk.push((id, seq.to_vec()));
        if chunk.len() >= 4096 {
            flush(&mut chunk, &mut kept, sink);
            progress(bytes_done);
            if let Some(lim) = limit && kept >= lim {
                done = true;
                break;
            }
        }
    }
    if !done {
        flush(&mut chunk, &mut kept, sink);
    }
    progress(bytes_done);
    if let Some(lim) = limit && kept > lim {
        kept = lim;
    }

    IndexStats {
        kept,
        skipped_short,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum HeaderFormat {
    /// First whitespace token of the header, verbatim (default)
    Token,
    /// `id|seqid ... [organism]`; headers without `|` fall back to the first token
    Pipe,
}

pub fn parse_header(header: &str, format: HeaderFormat) -> (String, String, String) {
    match format {
        HeaderFormat::Token => (
            header.split_whitespace().next().unwrap_or("").to_string(),
            String::new(),
            String::new(),
        ),
        HeaderFormat::Pipe => match header.split_once('|') {
            Some((id_part, rest)) => {
                let accession = rest.split_whitespace().next().unwrap_or("").to_string();
                let organism = rest
                    .rsplit_once('[')
                    .and_then(|(_, tail)| tail.strip_suffix(']'))
                    .unwrap_or("")
                    .to_string();
                (id_part.trim().to_string(), accession, organism)
            }
            None => parse_header(header, HeaderFormat::Token),
        },
    }
}

/// Borrowed view of an on-disk postings table (mmap region): u32 key, u32 ref id,
/// sorted by key. Used for the band postings.
pub struct PostingsTable<'a> {
    pub data: &'a [u8],
}

impl<'a> PostingsTable<'a> {
    pub fn len(&self) -> usize {
        self.data.len() / 8
    }
    pub fn key_at(&self, idx: usize) -> u32 {
        u32::from_le_bytes(self.data[idx * 8..idx * 8 + 4].try_into().unwrap())
    }
    pub fn ref_at(&self, idx: usize) -> u32 {
        u32::from_le_bytes(self.data[idx * 8 + 4..idx * 8 + 8].try_into().unwrap())
    }
    pub fn find(&self, key: u32) -> std::ops::Range<usize> {
        let n = self.len();
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.key_at(mid) < key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let start = lo;
        while lo < n && self.key_at(lo) == key {
            lo += 1;
        }
        start..lo
    }
}

/// Per-reference metadata beyond the hot path: strings only, deserialized
/// from the cold region only for hits that are actually reported.
#[derive(Serialize, Deserialize)]
struct ColdMeta {
    id: String,
    accession: String,
    organism: String,
}

/// Fixed-width hot records: sketch + protein length, random-access via stride.
pub struct HotTable<'a> {
    bytes: &'a [u8],
    stride: usize,
}

impl<'a> HotTable<'a> {
    fn len(&self) -> usize {
        self.bytes.len() / self.stride
    }
    fn get(&self, i: usize) -> Option<(&'a [u8], u32)> {
        if i * self.stride + self.stride > self.bytes.len() {
            return None;
        }
        let r = &self.bytes[i * self.stride..(i + 1) * self.stride];
        let sketch = &r[..SKETCH_BIN];
        let length = u32::from_le_bytes(r[SKETCH_BIN..SKETCH_BIN + 4].try_into().unwrap());
        Some((sketch, length))
    }
}

/// Cold-region record boundaries, u32 LE each, `n_hits + 1` entries:
/// record *i* lives in `cold[off[i]..off[i + 1]]`.
pub struct OffsetTable<'a> {
    bytes: &'a [u8],
}

impl<'a> OffsetTable<'a> {
    fn len(&self) -> usize {
        self.bytes.len() / 4
    }
    fn get(&self, i: usize) -> Option<u32> {
        self.bytes
            .get(i * 4..i * 4 + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    }
}

/// mmap-backed v10 index: header fields plus slices into the mapping.
pub struct MmapIndex {
    mmap: Mmap,
    hot_off: usize,
    off_off: usize,
    bands_off: usize,
    cold_off: usize,
    n_hits: usize,
    stride: usize,
    bands_len: usize,
}

pub fn load_mmap_index(path: &Path) -> std::io::Result<MmapIndex> {
    let file = File::open(path)?;
    // SAFETY: read-only mapping of a completed index file. The handle stays
    // open for the Mmap's lifetime and nothing truncates or rewrites the path
    // while the index is in use.
    let mmap = unsafe { Mmap::map(&file)? };
    let _ = mmap.advise(memmap2::Advice::Random);
    let bytes: &[u8] = &mmap;
    let bad = |what: String| std::io::Error::new(std::io::ErrorKind::InvalidData, what);
    if bytes.len() < HEADER_LEN {
        return Err(bad("truncated header".to_string()));
    }
    if &bytes[..4] != MAGIC {
        return Err(bad("bad magic".to_string()));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(bad(format!(
            "unsupported version {version} (expected {VERSION}: re-index your references)"
        )));
    }
    let n_hits = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let stride = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    if stride != HOT_STRIDE {
        return Err(bad("unexpected hot stride".to_string()));
    }
    let hot_off = HEADER_LEN;
    let off_off = hot_off + n_hits.saturating_mul(stride);
    let bands_off = off_off + (n_hits + 1) * 4;
    if bands_off + 8 > bytes.len() {
        return Err(bad("truncated bands table".to_string()));
    }
    let bands_len = u64::from_le_bytes(bytes[bands_off..bands_off + 8].try_into().unwrap()) as usize;
    let cold_off = bands_off + 8 + bands_len;
    if off_off > bytes.len() || cold_off > bytes.len() {
        return Err(bad("truncated hot/offset regions".to_string()));
    }
    Ok(MmapIndex {
        mmap,
        hot_off,
        off_off,
        bands_off,
        cold_off,
        n_hits,
        stride,
        bands_len,
    })
}

impl MmapIndex {
    pub fn n_hits(&self) -> usize {
        self.n_hits
    }

    /// Borrowed views into the mapping. Reborrow each batch if needed;
    /// all reads stay bounds-checked over the mapped bytes.
    pub fn views(&self) -> IndexViews<'_> {
        IndexViews {
            hot: HotTable {
                bytes: &self.mmap[self.hot_off..self.hot_off + self.n_hits * self.stride],
                stride: self.stride,
            },
            offs: OffsetTable {
                bytes: &self.mmap[self.off_off..self.off_off + (self.n_hits + 1) * 4],
            },
            bands: PostingsTable {
                data: &self.mmap[self.bands_off + 8..self.bands_off + 8 + self.bands_len],
            },
            cold: &self.mmap[self.cold_off..],
        }
    }
}

pub struct IndexViews<'a> {
    hot: HotTable<'a>,
    offs: OffsetTable<'a>,
    bands: PostingsTable<'a>,
    cold: &'a [u8],
}

impl<'a> IndexViews<'a> {
    pub fn n_hits(&self) -> usize {
        self.hot.len().min(self.offs.len().saturating_sub(1))
    }

    pub fn bands(&self) -> &PostingsTable<'a> {
        &self.bands
    }

    pub fn sketch(&self, i: usize) -> Option<&'a [u8]> {
        self.hot.get(i).map(|(s, _)| s)
    }

    /// Deserialize one cold record (id, accession, organism) on demand.
    pub fn cold_meta(&self, i: usize) -> Option<(String, String, String)> {
        let start = self.offs.get(i)? as usize;
        let end = self.offs.get(i + 1)? as usize;
        let m: ColdMeta = postcard::from_bytes(&self.cold[start..end]).ok()?;
        Some((m.id, m.accession, m.organism))
    }
}

pub struct Shard {
    pub path: std::path::PathBuf,
    pub records: u64,
}

// Write shard streams of postcard RefHit records.
pub fn collect_shards(
    input: &Path,
    min_len: usize,
    limit: Option<usize>,
    format: HeaderFormat,
    shard_dir: &Path,
    n_shards: usize,
    progress: &mut dyn FnMut(usize),
) -> std::io::Result<(Vec<Shard>, IndexStats)> {
    use std::io::Write;
    std::fs::create_dir_all(shard_dir)?;
    let mut shards: Vec<Shard> = (0..n_shards)
        .map(|i| Shard {
            path: shard_dir.join(format!("hits-{i}.tmp")),
            records: 0,
        })
        .collect();
    let mut files: Vec<std::io::BufWriter<std::fs::File>> = shards
        .iter()
        .map(|s| Ok(std::io::BufWriter::new(std::fs::File::create(&s.path)?)))
        .collect::<std::io::Result<_>>()?;
    let mut next = 0usize;
    let mut counts = vec![0u64; n_shards];
    let stats = collect_hits_sink(input, min_len, limit, format, progress, &mut |h| {
        let buf = postcard::to_allocvec(h).expect("serialize hit");
        let shard_idx = next;
        next = (next + 1) % files.len();
        let f = &mut files[shard_idx];
        f.write_all(&(buf.len() as u32).to_le_bytes()).unwrap();
        f.write_all(&buf).unwrap();
        counts[shard_idx] += 1;
    });
    for f in files.iter_mut() {
        f.flush()?;
    }
    for (s, &c) in shards.iter_mut().zip(counts.iter()) {
        s.records = c;
    }
    Ok((shards, stats))
}

// Splice shard record streams into one .plsh without re-serializing RefHit blobs.
// `progress` receives records decoded (out of `total_hits`).
pub fn aggregate_shards(
    out_path: &Path,
    shards: &[Shard],
    total_hits: u64,
    progress: &mut dyn FnMut(usize),
) -> std::io::Result<()> {
    use std::io::{Read, Write};
    let n = total_hits as usize;

    // Pass 1: rebuild hot/offsets/bands/cold bytes from the shard records.
    let mut hot = Vec::with_capacity(n * HOT_STRIDE);
    let mut cold = Vec::new();
    let mut offs: Vec<u32> = Vec::with_capacity(n + 1);
    offs.push(0u32);
    let mut band_postings: Vec<(u32, u32)> = Vec::with_capacity(n * crate::sketch::BANDS);
    let mut seen = 0u32;
    for s in shards {
        let mut r = std::io::BufReader::new(std::fs::File::open(&s.path)?);
        loop {
            let mut l = [0u8; 4];
            match r.read_exact(&mut l) {
                Ok(()) => {
                    let len = u32::from_le_bytes(l) as usize;
                    let mut buf = vec![0u8; len];
                    r.read_exact(&mut buf)?;
                    let hit: RefHit =
                        postcard::from_bytes(&buf).map_err(|e| {
                            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                        })?;
                    hot.extend_from_slice(&hit.sketch);
                    hot.extend_from_slice(&hit.length.to_le_bytes());

                    let meta = ColdMeta {
                        id: hit.id.clone(),
                        accession: hit.accession.to_string(),
                        organism: hit.organism.clone(),
                    };
                    let cbuf = postcard::to_allocvec(&meta).expect("serialize cold meta");
                    let next_off = offs[seen as usize] + cbuf.len() as u32;
                    if next_off as u64 > u32::MAX as u64 {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "cold region exceeds 4 GiB",
                        ));
                    }
                    cold.extend_from_slice(&cbuf);
                    offs.push(next_off);

                    for k in crate::sketch::band_keys(&hit.sketch) {
                        band_postings.push((k, seen));
                    }
                    seen += 1;
                    if seen.is_multiple_of(4096) {
                        progress(seen as usize);
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
        }
    }
    progress(seen as usize);

    let mut w = std::io::BufWriter::new(std::fs::File::create(out_path)?);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&total_hits.to_le_bytes())?;
    w.write_all(&(HOT_STRIDE as u32).to_le_bytes())?;
    w.write_all(&hot)?;
    for &o in &offs {
        w.write_all(&o.to_le_bytes())?;
    }
    {
        use rayon::prelude::*;
        band_postings.par_sort_unstable();
    }
    write_postings(&mut w, &band_postings)?;
    drop(band_postings);
    w.write_all(&cold)?;
    w.flush()
}

fn write_postings<W: Write>(w: &mut W, postings: &[(u32, u32)]) -> std::io::Result<()> {
    w.write_all(&((postings.len() * 8) as u64).to_le_bytes())?;
    for &(k, i) in postings {
        w.write_all(&k.to_le_bytes())?;
        w.write_all(&i.to_le_bytes())?;
    }
    Ok(())
}

pub fn write_index(
    path: &Path,
    hits: &[RefHit],
    progress: &mut dyn FnMut(usize),
) -> std::io::Result<()> {
    let n = hits.len();

    // Hot region: fixed-width sketch + length records, straight to disk.
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(n as u64).to_le_bytes())?;
    w.write_all(&(HOT_STRIDE as u32).to_le_bytes())?;
    for (i, h) in hits.iter().enumerate() {
        w.write_all(&h.sketch)?;
        w.write_all(&h.length.to_le_bytes())?;
        if i.is_multiple_of(4096) {
            progress(i);
        }
    }

    // Cold region with per-record offsets (n+1 entries, last = total).
    let mut cold: Vec<u8> = Vec::new();
    let mut offs: Vec<u32> = Vec::with_capacity(n + 1);
    offs.push(0u32);
    for h in hits {
        let meta = ColdMeta {
            id: h.id.clone(),
            accession: h.accession.to_string(),
            organism: h.organism.clone(),
        };
        let cbuf = postcard::to_allocvec(&meta).expect("serialize cold meta");
        let off = offs[offs.len() - 1] + cbuf.len() as u32;
        if off as u64 > u32::MAX as u64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "cold region exceeds 4 GiB",
            ));
        }
        cold.extend_from_slice(&cbuf);
        offs.push(off);
    }
    for &o in &offs {
        w.write_all(&o.to_le_bytes())?;
    }

    let mut band_postings: Vec<(u32, u32)> = Vec::with_capacity(n * crate::sketch::BANDS);
    for (i, h) in hits.iter().enumerate() {
        for k in crate::sketch::band_keys(&h.sketch) {
            band_postings.push((k, i as u32));
        }
    }
    band_postings.sort_unstable();
    write_postings(&mut w, &band_postings)?;
    w.write_all(&cold)?;
    progress(n);
    w.flush()
}

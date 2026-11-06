pub use indicatif::HumanDuration;
use indicatif::{ProgressBar, ProgressStyle};

/// One-line progress bar driven by absolute counts (bytes or items).
///
/// Draws to stderr and auto-hides when stderr isn't a real terminal (piped,
/// `TERM=dumb`, etc.), so stdout data (e.g. the search TSV) is never touched
/// and log files never see escape codes. Callers should check `is_hidden()`
/// to fall back to plain progress lines in that case.
pub struct Progress {
    bar: ProgressBar,
    goal: usize,
}

impl Progress {
    /// `template` is an indicatif template string, e.g.
    /// `"{bar:40} {pos}/{len} refs hashed (eta {eta})"` — see indicatif's
    /// docs for the full placeholder list (`bar`, `pos`, `len`, `bytes`,
    /// `total_bytes`, `eta`, `msg`, ...).
    pub fn start(goal: usize, template: &str) -> Self {
        let bar = if goal == 0 {
            ProgressBar::hidden()
        } else {
            ProgressBar::new(goal as u64)
        };
        bar.set_style(
            ProgressStyle::with_template(template)
                .expect("valid progress template")
                .progress_chars("##."),
        );
        Progress { bar, goal }
    }

    pub fn is_hidden(&self) -> bool {
        self.bar.is_hidden()
    }

    /// Move the bar to an absolute position.
    pub fn set(&mut self, done: usize) {
        self.bar.set_position(done as u64);
    }

    pub fn finish(&mut self) {
        self.bar.set_position(self.goal as u64);
        self.bar.finish_and_clear();
    }
}

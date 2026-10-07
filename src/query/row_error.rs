//! A per-thread "first data error" slot for row expressions.
//!
//! `physical::evaluate_expression` returns a plain `Value`, with no error
//! channel; structural mistakes are caught before execution by
//! `query::typecheck`. Some problems only show up in the data itself, e.g.
//! a spectrum whose m/z values aren't sorted. A function that finds one
//! records it here and returns `NULL`, and the executor that evaluated the
//! expression -- on the same thread -- calls `take()` after its loop and
//! fails the statement with that message. Executors `clear()` before they
//! start, so a stale error can never leak into another statement.

use std::cell::RefCell;

thread_local! {
    static FIRST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Records `message` unless an earlier error is already pending.
pub fn record(message: String) {
    FIRST_ERROR.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(message);
        }
    });
}

/// The pending error, if any, leaving the slot empty.
pub fn take() -> Option<String> {
    FIRST_ERROR.with(|slot| slot.borrow_mut().take())
}

pub fn clear() {
    let _ = take();
}

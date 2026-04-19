use coro::{Coro, suspend};
use std::cell::Cell;
use std::panic::AssertUnwindSafe;

/// Sets `self.0` to `true` when dropped.
struct DropRecorder<'a>(&'a Cell<bool>);

impl Drop for DropRecorder<'_> {
  fn drop(&mut self) {
    self.0.set(true);
  }
}

#[test]
fn no_unwind_when_not_started() {
  let dropped = Cell::new(false);
  Coro::new(|| {
    let _rec = DropRecorder(&dropped);
  });
  assert!(!dropped.get());
}

#[test]
fn unwind_when_finished() {
  let dropped = Cell::new(false);
  let mut coro = Coro::new(|| {
    let _rec = DropRecorder(&dropped);
    suspend();
    123
  });
  coro.resume();
  coro.resume();
  assert!(coro.has_finished());
  assert!(dropped.get());
}

#[test]
fn unwind_on_drop_before_finished() {
  let dropped = Cell::new(false);
  let mut coro = Coro::new(|| {
    let _rec = DropRecorder(&dropped);
    suspend();
  });
  coro.resume();
  assert!(!coro.has_finished());
  drop(coro);
  assert!(dropped.get());
}

#[test]
fn unwind_unfinished_inner_when_outer_finishes() {
  let inner_dropped = Cell::new(false);
  let inner_dropped_ref = AssertUnwindSafe(&inner_dropped);
  let mut outer = Coro::new(|| {
    let mut inner = Coro::new(|| {
      let _rec = DropRecorder(*inner_dropped_ref);
      suspend();
    });
    inner.resume();
  });
  outer.resume();
  assert!(inner_dropped.get());
  drop(outer);
}

#[test]
fn do_not_unwind_unfinished_inner_when_outer_finishes_if_decl_outside() {
  let inner_dropped = Cell::new(false);
  let inner_dropped_ref = AssertUnwindSafe(&inner_dropped);
  let mut inner = Coro::new(|| {
    let _rec = DropRecorder(*inner_dropped_ref);
    suspend();
  });
  Coro::new(|| inner.resume()).resume();
  assert!(!inner_dropped.get());
}

#[test]
fn unwind_unfinished_inner_on_outer_drop_before_finished() {
  let inner_dropped = Cell::new(false);
  let inner_dropped_ref = AssertUnwindSafe(&inner_dropped);
  let mut outer = Coro::new(move || {
    let mut inner = Coro::new(|| {
      let _rec = DropRecorder(*inner_dropped_ref);
      suspend();
    });
    inner.resume();
    // The only difference between `unwind_unfinished_inner_when_outer_finishes`
    // and this test is that here we call `suspend` to drop the outer subroutine
    // before it completely finishes execution.
    suspend();
  });
  outer.resume();
  drop(outer);
  assert!(inner_dropped.get());
}

#[test]
fn unwind_on_panic() {
  let dropped = Cell::new(false);
  let dropped_ref = AssertUnwindSafe(&dropped);
  let res = std::panic::catch_unwind(move || {
    Coro::new(|| {
      let _rec = DropRecorder(*dropped_ref);
      panic!("help");
    })
    .resume()
  });
  assert!(res.is_err());
  assert!(dropped.get());
}

#[test]
fn nested_unwinds_on_inner_panic() {
  let inner_dropped = Cell::new(false);
  let outer_dropped = Cell::new(false);
  let inner_dropped_ref = AssertUnwindSafe(&inner_dropped);
  let outer_dropped_ref = AssertUnwindSafe(&outer_dropped);
  let mut inner = Coro::new(|| {
    let _rec = DropRecorder(*inner_dropped_ref);
    panic!("inner");
  });
  let res = std::panic::catch_unwind(move || {
    Coro::new(|| {
      let _rec = DropRecorder(*outer_dropped_ref);
      inner.resume();
    })
    .resume()
  });
  assert!(res.is_err());
  assert!(inner_dropped.get());
  assert!(outer_dropped.get());
}

#[test]
fn nested_unwinds_on_outer_panic() {
  let inner_dropped = Cell::new(false);
  let outer_dropped = Cell::new(false);
  let inner_dropped_ref = AssertUnwindSafe(&inner_dropped);
  let outer_dropped_ref = AssertUnwindSafe(&outer_dropped);
  let mut inner = Coro::new(|| {
    let _rec = DropRecorder(*inner_dropped_ref);
    suspend();
  });
  let res = std::panic::catch_unwind(move || {
    Coro::new(|| {
      let _rec = DropRecorder(*outer_dropped_ref);
      inner.resume();
      panic!("outer");
    })
    .resume()
  });
  assert!(res.is_err());
  println!("outer coroutine panicked, and caught it: {res:?}");
  assert!(inner_dropped.get());
  assert!(outer_dropped.get());
}

#[test]
fn nested_unwinds_within_coroutine() {
  Coro::new(nested_unwinds_on_outer_panic).resume();
}

#[test]
fn panic_with_payload() {
  let res = std::panic::catch_unwind(|| {
    Coro::new(|| {
      panic!("foo");
    })
    .resume()
  });
  println!("{:?}", res.err());
}

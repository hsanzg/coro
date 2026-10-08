mod common;

use common::*;
use core::cell::Cell;
use core::panic::AssertUnwindSafe;
use coro::{Coro, in_coro, suspend};
use std::panic::{catch_unwind, resume_unwind};
use std::thread::panicking;

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
  let _ = new_coro(|| {
    let _rec = DropRecorder(&dropped);
  });
  assert!(!dropped.get());
}

#[test]
fn unwind_when_finished() {
  let dropped = Cell::new(false);
  let mut coro = new_coro(|| {
    let _rec = DropRecorder(&dropped);
    suspend();
    123
  });
  coro.resume();
  assert!(!dropped.get());
  coro.resume();
  assert!(dropped.get());
}

#[test]
fn unwind_on_drop_before_finished() {
  let dropped = Cell::new(false);
  let mut coro = new_coro(|| {
    let _rec = DropRecorder(&dropped);
    suspend();
  });
  coro.resume();
  assert!(!dropped.get());
  drop(coro);
  assert!(dropped.get());
  assert!(!in_coro());
}

#[test]
fn unwind_when_finished_within_pausing_coro() {
  let inner_dropped = Cell::new(false);
  let mut inner = new_coro(|| {
    let _rec = DropRecorder(&inner_dropped);
    suspend();
  });
  let mut outer = new_coro(|| {
    inner.resume();
    assert!(!inner_dropped.get());
    suspend();
    assert!(!inner_dropped.get());
    inner.resume();
    assert!(inner_dropped.get());
    assert!(in_coro());
  });
  outer.resume();
  outer.resume();
  assert!(!in_coro());
}

#[test]
fn unwind_unfinished_inner_when_outer_finishes() {
  let inner_dropped = Cell::new(false);
  let mut outer = new_coro(|| {
    let mut inner = new_coro(|| {
      let _rec = DropRecorder(&inner_dropped);
      suspend();
    });
    inner.resume();
  });
  outer.resume();
  assert!(inner_dropped.get());
  assert!(!in_coro());
}

#[test]
fn do_not_unwind_unfinished_inner_when_it_outlives_outer() {
  let inner_dropped = Cell::new(false);
  let mut inner = new_coro(|| {
    let _rec = DropRecorder(&inner_dropped);
    suspend();
  });
  let mut outer = new_coro(|| {
    inner.resume();
  });
  outer.resume();
  assert!(!inner_dropped.get());
  drop(outer);
  assert!(!inner_dropped.get());
  drop(inner);
  assert!(inner_dropped.get());
  assert!(!in_coro());
}

#[test]
fn unwind_unfinished_inner_on_outer_drop_before_finished() {
  let inner_dropped = Cell::new(false);
  let mut outer = new_coro(|| {
    let mut inner = new_coro(|| {
      let _rec = DropRecorder(&inner_dropped);
      suspend();
    });
    inner.resume();
    // The only difference between `unwind_unfinished_inner_when_outer_finishes`
    // and this test is that here we call `suspend` to drop the
    // outer subroutine before it completely finishes execution.
    suspend();
  });
  outer.resume();
  assert!(!inner_dropped.get());
  drop(outer);
  assert!(inner_dropped.get());
  assert!(!in_coro());
}

#[test]
fn user_unwind() {
  let dropped = Cell::new(false);
  let dropped_ref = AssertUnwindSafe(&dropped);
  let res = catch_unwind(move || {
    let mut coro = new_coro(|| {
      let _rec = DropRecorder(*dropped_ref);
      resume_unwind(Box::new(()));
    });
    coro.resume();
    assert_complete(&coro, true);
  });
  assert!(res.is_err());
  assert!(dropped.get());
  assert!(!in_coro());
}

/// On drop, checks that the current thread is unwinding
/// because of a panic.
struct PanicObserver;

impl Drop for PanicObserver {
  fn drop(&mut self) {
    assert!(panicking());
  }
}

#[test]
fn unwind_on_panic() {
  let dropped = Cell::new(false);
  let dropped_ref = AssertUnwindSafe(&dropped);
  let res = catch_unwind(move || {
    new_coro(|| {
      let _rec = DropRecorder(*dropped_ref);
      let _obs = PanicObserver;
      panic!("help");
    })
    .resume()
  });
  assert_eq!(
    res.expect_err("coro must panic").downcast_ref(),
    Some(&"help")
  );
  assert!(dropped.get());
  assert!(!in_coro());
}

#[test]
fn unwind_on_panic_after_many_suspends_from_recursive_fn() {
  const COUNTDOWN: u32 = 100;
  let mut coro = new_coro(|| {
    fn tick(t: u32) {
      let _obs = PanicObserver;
      if t == 0 {
        panic!("boom");
      }
      suspend();
      tick(t - 1)
    }
    tick(COUNTDOWN);
  });
  for _ in 0..COUNTDOWN {
    coro.resume();
  }
  let mut coro_ref = AssertUnwindSafe(&mut coro);
  let res = catch_unwind(move || coro_ref.resume());
  assert_eq!(
    res.expect_err("coro must panic").downcast_ref(),
    Some(&"boom")
  );
  assert_complete(&coro, true);
  assert!(!in_coro());
}

#[test]
fn nested_unwinds_on_inner_panic() {
  let inner_dropped = Cell::new(false);
  let outer_dropped = Cell::new(false);
  let inner_dropped_ref = AssertUnwindSafe(&inner_dropped);
  let outer_dropped_ref = AssertUnwindSafe(&outer_dropped);
  let mut inner = new_coro(|| {
    let _rec = DropRecorder(*inner_dropped_ref);
    let _obs = PanicObserver;
    panic!("inner");
  });
  let res = catch_unwind(move || {
    new_coro(|| {
      let _rec = DropRecorder(*outer_dropped_ref);
      let _obs = PanicObserver;
      inner.resume();
    })
    .resume()
  });
  assert_eq!(
    res.expect_err("inner must panic").downcast_ref(),
    Some(&"inner")
  );
  assert!(inner_dropped.get());
  assert!(outer_dropped.get());
  assert!(!in_coro());
}

#[test]
fn doubly_nested_unwinds_on_inner_panic() {
  let inner_dropped = Cell::new(false);
  let middle_dropped = Cell::new(false);
  let outer_dropped = Cell::new(false);
  let inner_dropped_ref = AssertUnwindSafe(&inner_dropped);
  let middle_dropped_ref = AssertUnwindSafe(&middle_dropped);
  let outer_dropped_ref = AssertUnwindSafe(&outer_dropped);
  let mut outer = new_coro(|| {
    let _rec = DropRecorder(*outer_dropped_ref);
    let _obs = PanicObserver;
    new_coro(|| {
      let _rec = DropRecorder(*middle_dropped_ref);
      let _obs = PanicObserver;
      new_coro(|| {
        let _rec = DropRecorder(*inner_dropped_ref);
        let _obs = PanicObserver;
        panic!("inner");
      })
      .resume();
    })
    .resume();
  });
  let mut outer_ref = AssertUnwindSafe(&mut outer);
  let res = catch_unwind(move || outer_ref.resume());
  assert_eq!(
    res.expect_err("inner must panic").downcast_ref(),
    Some(&"inner")
  );
  assert_complete(&outer, true);
  assert!(inner_dropped.get());
  assert!(middle_dropped.get());
  assert!(outer_dropped.get());
  assert!(!in_coro());
}

#[test]
fn nested_unwinds_on_outer_panic() {
  let inner_dropped = Cell::new(false);
  let outer_dropped = Cell::new(false);
  let inner_dropped_ref = AssertUnwindSafe(&inner_dropped);
  let outer_dropped_ref = AssertUnwindSafe(&outer_dropped);
  let mut inner = new_coro(|| {
    let _rec = DropRecorder(*inner_dropped_ref);
    let _obs = PanicObserver;
    suspend();
  });
  let res = catch_unwind(move || {
    new_coro(|| {
      let _rec = DropRecorder(*outer_dropped_ref);
      let _obs = PanicObserver;
      inner.resume();
      panic!("outer");
    })
    .resume()
  });
  assert_eq!(
    res.expect_err("outer must panic").downcast_ref(),
    Some(&"outer")
  );
  assert!(inner_dropped.get());
  assert!(outer_dropped.get());
  assert!(!in_coro());
}

#[test]
fn doubly_nested_unwinds_on_outer_panic() {
  let inner_dropped = Cell::new(false);
  let middle_dropped = Cell::new(false);
  let outer_dropped = Cell::new(false);
  let inner_dropped_ref = AssertUnwindSafe(&inner_dropped);
  let middle_dropped_ref = AssertUnwindSafe(&middle_dropped);
  let outer_dropped_ref = AssertUnwindSafe(&outer_dropped);
  let mut outer = new_coro(|| {
    let _rec = DropRecorder(*outer_dropped_ref);
    let _obs = PanicObserver;
    // Note that the stack of the inner coroutines is not unwound
    // as a direct consequence of the panic, but rather because
    // their destructor is called. It is therefore incorrect to
    // attach a `PanicObserver` to them.
    new_coro(|| {
      let _rec = DropRecorder(*middle_dropped_ref);
      new_coro(|| {
        let _rec = DropRecorder(*inner_dropped_ref);
        suspend();
      })
      .resume();
      suspend();
    })
    .resume();
    panic!("outer");
  });
  let res = catch_unwind(move || outer.resume());
  assert_eq!(
    res.expect_err("outer must panic").downcast_ref(),
    Some(&"outer")
  );
  assert!(inner_dropped.get());
  assert!(middle_dropped.get());
  assert!(outer_dropped.get());
  assert!(!in_coro());
}

#[test]
fn catch_panic_unwind_within_coro() {
  let mut coro = new_coro(|| {
    let res = catch_unwind(|| panic!("help"));
    assert!(res.is_err());
    assert!(in_coro());
  });
  coro.resume();
  assert_complete(&coro, false);
  assert!(!in_coro());
}

#[test]
fn catch_unwind_of_inner_coro_within_coro() {
  let thrower_dropped = Cell::new(false);
  let catcher_dropped = Cell::new(false);
  let thrower_dropped_ref = AssertUnwindSafe(&thrower_dropped);
  let mut catcher = new_coro(|| {
    let _rec = DropRecorder(&catcher_dropped);
    let mut thrower = new_coro(|| {
      let _rec = DropRecorder(*thrower_dropped_ref);
      panic!("bet you can't catch me!");
    });
    let mut thrower_ref = AssertUnwindSafe(&mut thrower);
    let res = catch_unwind(move || thrower_ref.resume());
    assert_eq!(
      res.expect_err("thrower must panic").downcast_ref(),
      Some(&"bet you can't catch me!")
    );
    assert_complete(&thrower, true);
    assert!(thrower_dropped.get());
    assert!(in_coro());
    suspend();
    assert!(in_coro());
  });
  catcher.resume();
  assert_paused(&catcher);
  assert!(!catcher_dropped.get());
  assert!(!in_coro());
  catcher.resume();
  assert_complete(&catcher, false);
  assert!(catcher_dropped.get());
  assert!(!in_coro());
}

/// Calls [`suspend`] when dropped.
struct SuspendOnDrop;
impl Drop for SuspendOnDrop {
  fn drop(&mut self) {
    println!("suspend from drop");
    suspend();
  }
}

#[test]
#[should_panic]
// #[ignore = "aborts the test process; uncomment and test manually"]
// todo: `_panics` instead of `_aborts`
fn suspend_from_paused_coro_being_dropped_aborts() {
  new_coro(|| {
    let _drop = SuspendOnDrop;
    suspend();
  })
  .resume();
}

#[test]
#[ignore = "aborts the test process; uncomment and test manually"]
fn suspend_from_coro_being_unwound_due_to_panic_aborts() {
  new_coro(|| {
    let _drop = SuspendOnDrop;
    panic!();
  })
  .resume();
}

// todo: add `suspend_from_dropping_coro_panics`.
// todo: add `resume_from_dropping_coro`.
// todo: add `resume_another_coro_when_being_dropped` (due to panic and due to drop).

#![feature(coroutine_trait)]

mod common;

use common::*;
use core::ops::{Coroutine, CoroutineState};
use core::pin::Pin;
use coro::{Coro, suspend};

#[test]
#[should_panic]
fn suspend_from_main_program_panics() {
  suspend();
}

#[test]
fn resume_and_complete_with_unit_return_value() {
  let mut coro = new_coro(|| {});
  assert_unstarted(&coro);

  assert_eq!(coro.resume(), CoroutineState::Complete(()));
  assert_complete(&coro, false);
}

#[test]
fn resume_and_complete_with_nonunit_return_value() {
  let mut coro = new_coro(|| 123);
  assert_eq!(coro.resume(), CoroutineState::Complete(123));
  assert_complete(&coro, false);
}

#[test]
#[should_panic]
fn resume_after_finishing_panics() {
  let mut coro = new_coro(|| {});
  coro.resume();
  coro.resume(); // panics.
}

#[test]
fn resume_suspend_resume() {
  let mut coro = new_coro(|| {
    let ret = 456;
    suspend();
    ret
  });
  assert_eq!(coro.resume(), CoroutineState::Yielded(()));
  assert_paused(&coro);

  assert_eq!(coro.resume(), CoroutineState::Complete(456));
  assert_complete(&coro, false);
}

#[test]
fn resume_via_trait_method() {
  let mut coro = new_coro(|| {
    suspend();
    4
  });
  assert_eq!(Pin::new(&mut coro).resume(()), CoroutineState::Yielded(()));
  assert_paused(&coro);
  assert_eq!(Pin::new(&mut coro).resume(()), CoroutineState::Complete(4));
  assert_complete(&coro, false);
}

#[test]
fn many_control_transfers() {
  const COUNT: u32 = 100;
  let mut coro = new_coro(|| {
    for _ in 0..COUNT {
      suspend();
    }
    'x'
  });
  for _ in 0..COUNT {
    assert_eq!(coro.resume(), CoroutineState::Yielded(()));
    assert_paused(&coro);
  }
  assert_eq!(coro.resume(), CoroutineState::Complete('x'));
  assert_complete(&coro, false);
}

#[test]
fn resume_after_move() {
  let mut coro = new_coro(|| {
    suspend();
    0xcafe
  });
  coro.resume();

  #[inline(never)]
  fn resume<F: FnOnce() -> i32 + Unpin>(mut coro: Coro<i32, F>) {
    assert_eq!(coro.resume(), CoroutineState::Complete(0xcafe));
    assert_complete(&coro, false);
  }
  resume(coro);
}

#[test]
fn nested_coros() {
  let mut inner = new_coro(|| {
    suspend();
    "inner"
  });
  let mut outer = new_coro(|| {
    assert_unstarted(&inner);

    assert_eq!(inner.resume(), CoroutineState::Yielded(()));
    assert_paused(&inner);
    suspend();
    assert_paused(&inner);

    assert_eq!(inner.resume(), CoroutineState::Complete("inner"));
    assert_complete(&inner, false);
    suspend();
    assert_complete(&inner, false);
    "outer"
  });

  for _ in 0..2 {
    assert_eq!(outer.resume(), CoroutineState::Yielded(()));
    assert_paused(&outer);
  }
  assert_eq!(outer.resume(), CoroutineState::Complete("outer"));
  assert_complete(&outer, false);
}

#[test]
fn triply_nested_coros() {
  let mut inner = new_coro(|| {
    suspend();
    "inner"
  });
  let mut middle = new_coro(|| {
    assert_eq!(inner.resume(), CoroutineState::Yielded(()));
    assert_paused(&inner);
    suspend();
    assert_paused(&inner);

    assert_eq!(inner.resume(), CoroutineState::Complete("inner"));
    assert_complete(&inner, false);
    suspend();
    assert_complete(&inner, false);
    "middle"
  });
  let mut outer = new_coro(|| {
    for _ in 0..2 {
      assert_eq!(middle.resume(), CoroutineState::Yielded(()));
      assert_paused(&middle);
      suspend();
      assert_paused(&middle);
    }

    assert_eq!(middle.resume(), CoroutineState::Complete("middle"));
    assert_complete(&middle, false);
    suspend();
    assert_complete(&middle, false);
    "outer"
  });

  for _ in 0..3 {
    assert_eq!(outer.resume(), CoroutineState::Yielded(()));
    assert_paused(&outer);
  }
  assert_eq!(outer.resume(), CoroutineState::Complete("outer"));
  assert_complete(&outer, false);
}

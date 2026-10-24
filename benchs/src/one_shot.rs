#![feature(coroutine_trait)]

use coro::{Coro, suspend};
use corosensei::{Coroutine, CoroutineResult};
use gungraun::prelude::*;
use std::hint::black_box;
use std::ops::CoroutineState;

fn coro_switch_setup() -> Coro<()> {
  Coro::new(|| suspend())
}

#[library_benchmark]
#[bench::with_setup(setup = coro_switch_setup)]
fn coro_switch(mut coro: Coro<()>) -> Coro<()> {
  black_box(coro.resume());
  coro
}

fn corosensei_switch_setup() -> Coroutine<(), (), ()> {
  Coroutine::new(|y, _| y.suspend(()))
}

#[library_benchmark]
#[bench::with_setup(setup = corosensei_switch_setup)]
fn corosensei_switch(mut coro: Coroutine<(), (), ()>) -> Coroutine<(), (), ()> {
  black_box(coro.resume(()));
  coro
}

library_benchmark_group!(
  name = switch_group,
  benchmarks = [coro_switch, corosensei_switch]
);

#[library_benchmark]
fn coro_call() -> CoroutineState<(), ()> {
  Coro::new(|| {}).resume()
}

#[library_benchmark]
fn corosensei_call() -> CoroutineResult<(), ()> {
  Coroutine::new(|_, _| {}).resume(())
}

library_benchmark_group!(name = call_group, benchmarks = [coro_call, corosensei_call]);

main!(library_benchmark_groups = [switch_group, call_group]);

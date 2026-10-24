use coro::{Coro, suspend};
use corosensei::{Coroutine, Yielder};
use criterion::{Criterion, criterion_group, criterion_main};

fn switch(c: &mut Criterion) {
  let mut group = c.benchmark_group("switch");

  let mut coro = Coro::new(|| {
    loop {
      suspend();
    }
  });
  group.bench_function("coro", |b| b.iter(|| coro.resume()));

  let mut coro = Coroutine::new(|yielder, ()| {
    loop {
      yielder.suspend(());
    }
  });
  group.bench_function("corosensei", |b| b.iter(|| coro.resume(())));
}

// todo: repeat this benchmark, but without stack pooling.
fn call(c: &mut Criterion) {
  let mut group = c.benchmark_group("call");

  group.bench_function("coro", |b| {
    b.iter(|| {
      Coro::new(|| {}).resume();
    })
  });

  group.bench_function("corosensei", |b| {
    b.iter(|| {
      Coroutine::new(|_yielder: &Yielder<(), ()>, ()| {}).resume(());
    })
  });
}

// todo: add `panic` and `paused_drop` benchmarks.

criterion_group!(benchs, switch, call);
criterion_main!(benchs);

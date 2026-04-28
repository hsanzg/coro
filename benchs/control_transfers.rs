/*use coro::{Coro, suspend};
use criterion::{Criterion, criterion_group, criterion_main};

fn control_transfers(c: &mut Criterion) {
  let mut coro = Coro::new(|| {
    loop {
      suspend();
    }
  });
  c.bench_function("switch", |b| b.iter(|| coro.resume()));
  // todo: do this benchmark, but without stack pooling.
  c.bench_function("call", |b| {
    b.iter(|| {
      Coro::new(|| {}).resume();
    })
  });
}

criterion_group!(benches, control_transfers);
criterion_main!(benches);
*/

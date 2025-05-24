# dd-sketchy



cargo build --bin add_operations
cargo build --bin merge_operations
cargo build --bin quantile_operations


sudo cargo flamegraph --bin add_operations -- ./target/debug/add_operations
sudo cargo flamegraph --bin merge_operations -- ./target/debug/merge_operations
sudo cargo flamegraph --bin quantile_operations -- ./target/debug/quantile_operations

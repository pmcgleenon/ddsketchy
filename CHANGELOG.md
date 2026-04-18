# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.6](https://github.com/pmcgleenon/ddsketchy/compare/v0.1.5...v0.1.6) - 2026-04-18

### Other

- expand crate-level docs, doctests, and docs CI ([#28](https://github.com/pmcgleenon/ddsketchy/pull/28))
- inline hot-path in add() and drop redundant checks ([#27](https://github.com/pmcgleenon/ddsketchy/pull/27))
- optimize merge with slice zip and copy_within ([#25](https://github.com/pmcgleenon/ddsketchy/pull/25))

## [0.1.5](https://github.com/pmcgleenon/ddsketchy/compare/v0.1.4...v0.1.5) - 2026-03-07

### Other

- python bindings for ddsketch ([#23](https://github.com/pmcgleenon/ddsketchy/pull/23))

## [0.1.4](https://github.com/pmcgleenon/ddsketchy/compare/v0.1.3...v0.1.4) - 2026-01-03

### Fixed

- MinIndexableValue to make relative error within bounds for small values

## [0.1.3](https://github.com/pmcgleenon/ddsketchy/compare/v0.1.2...v0.1.3) - 2025-12-11

### Other

- remove floating point operations

### Removed

- removed benchmark execution
- removed duplicated calculation

## [0.1.2](https://github.com/pmcgleenon/ddsketchy/compare/v0.1.1...v0.1.2) - 2025-12-07

### Other

- added dual store and expanded tests to validate results against datadog reference implementation
- *(deps)* update criterion requirement from 0.6 to 0.7


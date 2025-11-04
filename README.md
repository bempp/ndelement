# ndelement

[![DefElement verification](https://defelement.org/badges/ndelement.svg)](https://defelement.org/verification/ndelement.html)
[![crates.io](https://img.shields.io/crates/v/ndelement?color=blue)](https://crates.io/crates/ndelement)
[![docs.rs](https://img.shields.io/docsrs/ndelement?label=docs.rs)](https://docs.rs/ndelement/latest/ndelement/)
[![PyPI](https://img.shields.io/pypi/v/ndelement?color=blue&label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/ndelement/)

ndelement is an open-source library written in Rust that can be used to create n-dimensional finite elements.

## Using ndelement
### Rust
You can use the latest release of ndelement by adding the following to `[dependencies]` section of your Cargo.toml file:

```toml
ndelement = "0.3.0"
```

### Python
You can install the latest release of ndelement by running:

```bash
pip3 install ndelement
```

## Documentation
The latest documentation of the main branch of this repo is available at [bempp.github.io/ndelement/](https://bempp.github.io/ndelement/).

## Testing
The Rust functionality of the library can be tested by running:
```bash
cargo test
```

The Python functionality of the library can be tested by running:
```bash
python -m pytest python/test
```

## Examples
Examples of use can be found in the [examples folder](examples/).

## Getting help
Documentation of the latest release of ndelement can be found on [docs.rs](https://docs.rs/ndelement/latest/ndelement/).
Documentation of the latest development version of ndelement can be found at [bempp.github.io/ndelement/ndelement](https://bempp.github.io/ndelement/ndelement).

Errors in the library should be added to the [GitHub issue tracker](https://github.com/bempp/ndelement/issues).

Questions about the library and its use can be asked on the [Bempp Discourse](https://bempp.discourse.group).

## Licence
ndelement is licensed under a BSD 3-Clause licence. Full text of the licence can be found [here](LICENSE.md).

# ndelement

ndelement is an open-source library written in Rust that can be used to create n-dimensional finite elements.

## Using ndelement
### Rust
You can use the latest release of ndelement by adding the following to `[dependencies]` section of your Cargo.toml file:

```toml
ndelement = "0.1.1"
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
Errors in the library should be added to the [GitHub issue tracker](https://github.com/bempp/ndelement/issues).

Questions about the library and its use can be asked on the [Bempp Discourse](https://bempp.discourse.group).

## Licence
ndelement is licensed under a BSD 3-Clause licence. Full text of the licence can be found [here](LICENSE.md).

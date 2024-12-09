# Making a release

To make a new release of ndelement, follow the following steps:

0) If you are yet to make a release on your current computer, run `cargo login` and copy an API
   key from https://crates.io/me

1) Checkout the `main` branch and `git pull`, then checkout a new branch called `release-v[x].[y].[z]`
   (where `[x]`, `[y]`, and `[z]` are defined in the next step):
   ```bash
   git checkout main
   git pull
   git checkout -b release-v[x].[y].[z]
   ```

2) Update the version numbers in `Cargo.toml`, `pyproject.toml` and `README.md`.
   The version numbers have the format `[x].[y].[z]`. If you are releasing a major
   version, you should increment `[x]` and set `[y]` and `[z]` to 0.
   If you are releasing a minor version, you should increment `[y]` and set `[z]`
   to zero. If you are releasing a bugfix, you should increment `[z]`.

3) Run `cargo publish --dry-run` and fix any errors.

4) Commit your changes and push to GitHub, open a pull request to merge changes back into main, and merge the
   pull request.

5) [Create a release on GitHub](https://github.com/bempp/ndelement/releases/new) from the `main` branch.
   The release tag and title should be `v[x].[y].[z]` (where `[x]`, `[y]` and `[z]` are as in step 2).
   In the "Describe this release" box, you should bullet point the main changes since the last
   release.

6) Run `cargo publish`. This will push the new version to crates.io.
   Note: this cannot be undone, but you can use `cargo yank` to mark a version as unsuitable for use.

7) Open a pull request to `main` to update the version numbers in `Cargo.toml` and `pyproject.toml`
   to `[x].[y].[z]-dev`

8) Add the release to the next issue of [Scientific Computing in Rust Monthly](https://github.com/rust-scicomp/scientific-computing-in-rust-monthly)

# Fork release bootstrap

Release automation is fail-closed. The `Publish and Release` workflow validates
its checked-in configuration on every maintained-line push, but it skips both
Release Please and npm publication unless the repository Actions variable
`RELEASE_ENABLED` is exactly `true`.

The checked-in version contract is the fork prerelease line
`0.59.0-proliferate.N`. Release Please reads `release-please-config.json` and
`.release-please-manifest.json`; the workflow must not also set its
`release-type` input because that switches the action into a mode that ignores
both files. The manifest config also disables component-prefixed tags so the
fork baseline is `v0.59.0-proliferate.1`, not
`claude-agent-acp-v0.59.0-proliferate.1`.

## One-time activation

Complete these steps in order. Keep `RELEASE_ENABLED` unset until the last
step.

1. Have an authorized npm owner review the package contents and perform the
   first public publish of the exact current version,
   `@proliferate/claude-agent-acp@0.59.0-proliferate.1`. npm trusted publishing
   cannot create a package that does not already exist.
2. Create the matching `v0.59.0-proliferate.1` tag and GitHub prerelease at the
   exact commit used for that package. This gives Release Please an unambiguous
   fork baseline; do not reuse one of the legacy `v0.24.2-proliferate.N` tags.
3. In npm package settings, add a GitHub Actions trusted publisher for
   organization `proliferate-ai`, repository `claude-agent-acp`, workflow
   `publish.yml`, and environment `release`. Set **Allowed actions** to
   `npm publish`.
4. Create or select a fork-owned GitHub App, install it only where needed, and
   grant repository Contents, Issues, and Pull requests read/write access. Add
   its Client ID as the repository Actions variable `RELEASE_PLZ_CLIENT_ID` and
   its private key as the `release` environment secret
   `RELEASE_PLZ_APP_PRIVATE_KEY`.
5. Confirm the maintained branch and `release` environment protections match
   the repository's release-approval policy. The App token is intentionally
   used instead of `GITHUB_TOKEN` so its release PRs trigger normal CI.
6. Set the repository Actions variable `RELEASE_ENABLED` to `true`, then
   manually dispatch `Publish and Release` on `upstream/v0.59`. With no
   releasable commits after the baseline tag, the run should succeed without
   creating a PR, tag, release, or npm publication.

Manual dispatches may create or update a Release Please PR, but they cannot
create a GitHub release or publish to npm. Those mutations only occur on a push
to `upstream/v0.59`, and publication additionally verifies that the new release
tag points to the workflow's exact commit.

After activation, a conventional `fix` or `feat` commit advances the prerelease
counter (for example, `.1` to `.2`). Review and merge Release Please PRs through
the normal branch policy; never merge one while the npm trusted publisher or
release environment is unavailable.

To stop release mutations without changing the workflow, delete
`RELEASE_ENABLED` or set it to any value other than the lowercase string
`true`.

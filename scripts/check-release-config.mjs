import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

function jobBlock(workflow, job) {
  const lines = workflow.split("\n");
  const start = lines.indexOf(`  ${job}:`);
  assert.notEqual(start, -1, `missing ${job} job`);

  const nextJob = lines.findIndex(
    (line, index) => index > start && /^  [a-zA-Z0-9_-]+:$/.test(line),
  );
  return lines.slice(start, nextJob === -1 ? undefined : nextJob).join("\n");
}

function actionStep(job, action) {
  const lines = job.split("\n");
  const uses = lines.findIndex(
    (line) =>
      line.startsWith(`      - uses: ${action}@`) || line.startsWith(`        uses: ${action}@`),
  );
  assert.notEqual(uses, -1, `missing ${action} step`);

  let start = uses;
  while (start >= 0 && !/^      - /.test(lines[start])) {
    start -= 1;
  }
  assert.notEqual(start, -1, `missing start of ${action} step`);

  const end = lines.findIndex(
    (line, index) => index > start && (/^      - /.test(line) || /^ {0,6}\S/.test(line)),
  );
  return lines.slice(start, end === -1 ? undefined : end).join("\n");
}

function namedStep(job, name) {
  const lines = job.split("\n");
  const start = lines.indexOf(`      - name: ${name}`);
  assert.notEqual(start, -1, `missing ${name} step`);

  const end = lines.findIndex(
    (line, index) => index > start && (/^      - /.test(line) || /^ {0,6}\S/.test(line)),
  );
  return lines.slice(start, end === -1 ? undefined : end).join("\n");
}

function exactSetting(block, indentation, key) {
  const prefix = `${" ".repeat(indentation)}${key}:`;
  const matches = block.split("\n").filter((line) => line.startsWith(prefix));
  assert.equal(matches.length, 1, `expected exactly one ${key} setting`);
  return matches[0].slice(prefix.length).trim();
}

const [packageJson, packageLock, releaseConfig, releaseManifest, publishWorkflow] =
  await Promise.all([
    readJson("package.json"),
    readJson("package-lock.json"),
    readJson("release-please-config.json"),
    readJson(".release-please-manifest.json"),
    readFile(".github/workflows/publish.yml", "utf8"),
  ]);

const packageConfig = releaseConfig.packages?.["."];
assert.ok(packageConfig, "release-please must configure the root package");
assert.equal(packageConfig["release-type"], "node");
assert.equal(packageConfig["include-v-in-tag"], true);
assert.equal(packageConfig["include-component-in-tag"], false);
assert.equal(packageConfig.versioning, "prerelease");
assert.equal(packageConfig.prerelease, true);
assert.equal(packageConfig["prerelease-type"], "proliferate");

assert.equal(packageJson.name, "@proliferate/claude-agent-acp");
assert.equal(packageLock.name, packageJson.name);
assert.equal(packageLock.packages?.[""]?.name, packageJson.name);
assert.equal(packageLock.version, packageJson.version);
assert.equal(packageLock.packages?.[""]?.version, packageJson.version);
assert.equal(releaseManifest["."], packageJson.version);
assert.match(packageJson.version, /^0\.59\.0-proliferate\.\d+$/);

const releaseConfigJob = jobBlock(publishWorkflow, "release-config");
const releaseJob = jobBlock(publishWorkflow, "release-please");
const publishJob = jobBlock(publishWorkflow, "publish-npm");
const appTokenStep = actionStep(releaseJob, "actions/create-github-app-token");
const releaseActionStep = actionStep(releaseJob, "googleapis/release-please-action");
const verifyTagStep = namedStep(publishJob, "Verify release tag matches workflow commit");

assert.ok(
  releaseConfigJob.split("\n").includes("      - run: npm run check:release"),
  "release-config job must run the semantic validator",
);
assert.equal(exactSetting(releaseJob, 4, "needs"), "[release-config]");
assert.equal(
  exactSetting(releaseJob, 4, "if"),
  "${{ vars.RELEASE_ENABLED == 'true' && github.ref == 'refs/heads/upstream/v0.59' }}",
);
assert.equal(exactSetting(releaseJob, 4, "environment"), "release");
assert.equal(exactSetting(releaseJob, 4, "permissions"), "{}");
assert.equal(exactSetting(appTokenStep, 10, "client-id"), "${{ vars.RELEASE_PLZ_CLIENT_ID }}");
assert.equal(
  exactSetting(appTokenStep, 10, "private-key"),
  "${{ secrets.RELEASE_PLZ_APP_PRIVATE_KEY }}",
);
assert.equal(
  appTokenStep.split("\n").some((line) => /^ {10}app-id:/.test(line)),
  false,
  "deprecated app-id input must stay absent",
);
assert.equal(exactSetting(appTokenStep, 10, "permission-contents"), "write");
assert.equal(exactSetting(appTokenStep, 10, "permission-issues"), "write");
assert.equal(exactSetting(appTokenStep, 10, "permission-pull-requests"), "write");
assert.equal(
  exactSetting(releaseActionStep, 10, "token"),
  "${{ steps.generate-token.outputs.token }}",
);
assert.equal(exactSetting(releaseActionStep, 10, "target-branch"), "upstream/v0.59");
assert.equal(exactSetting(releaseActionStep, 10, "config-file"), "release-please-config.json");
assert.equal(exactSetting(releaseActionStep, 10, "manifest-file"), ".release-please-manifest.json");
assert.equal(
  exactSetting(releaseActionStep, 10, "skip-github-release"),
  "${{ github.event_name == 'workflow_dispatch' }}",
);
assert.equal(
  releaseActionStep.split("\n").some((line) => /^ {10}release-type:/.test(line)),
  false,
  "release-type input would bypass manifest mode",
);
assert.equal(exactSetting(publishJob, 4, "needs"), "[release-please]");
assert.equal(
  exactSetting(publishJob, 4, "if"),
  "${{ vars.RELEASE_ENABLED == 'true' && github.event_name == 'push' && github.ref == 'refs/heads/upstream/v0.59' && needs.release-please.outputs.release_created == 'true' }}",
);
assert.equal(exactSetting(publishJob, 4, "environment"), "release");
assert.ok(
  publishJob.split("\n").includes("      id-token: write"),
  "npm publish job must retain OIDC permission",
);
assert.equal(
  verifyTagStep,
  [
    "      - name: Verify release tag matches workflow commit",
    "        env:",
    "          RELEASE_TAG: ${{ needs.release-please.outputs.tag_name }}",
    "        run: |",
    '          if [[ -z "$RELEASE_TAG" ]] || ! git check-ref-format "refs/tags/$RELEASE_TAG"; then',
    '            echo "::error title=Invalid release tag::Release Please returned an empty or invalid tag."',
    "            exit 1",
    "          fi",
    '          git fetch --no-tags --depth=1 origin "refs/tags/$RELEASE_TAG:refs/tags/$RELEASE_TAG"',
    '          tag_commit="$(git rev-parse "refs/tags/$RELEASE_TAG^{commit}")"',
    '          if [[ "$tag_commit" != "$GITHUB_SHA" ]]; then',
    '            echo "::error title=Release commit mismatch::Tag $RELEASE_TAG points to $tag_commit, but this workflow is publishing $GITHUB_SHA."',
    "            exit 1",
    "          fi",
  ].join("\n"),
  "npm publish tag verification must stay fail-closed",
);

console.log("Release configuration is internally consistent and fail-closed.");

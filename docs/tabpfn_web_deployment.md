# TabPFN Web deployment

bochan Web treats TabPFN model weights as deployment assets. The public Web
runtime must not perform Prior Labs authentication or download checkpoints while
handling user requests.

## Production provisioning

1. Accept the required TabPFN license in the Prior Labs account used for deployment.
2. Store `TABPFN_TOKEN` in the deployment platform's secret manager.
3. Mount a persistent directory for TabPFN checkpoints and expose it as
   `TABPFN_MODEL_CACHE_DIR`.
4. Run the preload command in a deployment job or init container:

```bash
export TABPFN_TOKEN="..."
export TABPFN_MODEL_CACHE_DIR="/var/lib/bochan/tabpfn"
python -m bochan.serving.webapp.tabpfn_preload
```

The preload command downloads the official default TabPFN v3 classifier and
regressor checkpoints only. It does not download all experimental variants.
Browser authentication is disabled by default, so an automated deployment fails
clearly when the token/license is unavailable.

5. Start the normal bochan Web container with the same checkpoint directory
   mounted, preferably read-only. Do not expose `TABPFN_TOKEN` to the runtime
   container.

```bash
export TABPFN_MODEL_CACHE_DIR="/var/lib/bochan/tabpfn"
# start bochan Web normally
```

For stronger isolation, deny runtime egress to Prior Labs/Hugging Face at the
infrastructure or network-policy layer as well.

## Local interactive provisioning

For a developer workstation only, the preload command can explicitly allow the
TabPFN library's browser login flow:

```bash
python -m bochan.serving.webapp.tabpfn_preload --allow-browser-auth
```

## Runtime contract

When a real bochan Web request selects TabPFN, bochan checks that both required
default checkpoints are already present and non-empty before constructing the
estimator. If either checkpoint is missing, the request fails with a provisioning
error that points to the preload command. The Web process forces
`TABPFN_NO_BROWSER=1`, so browser authentication is never used as a runtime
fallback.

Explicitly injected, already-constructed estimators used by tests or advanced
programmatic integrations are not subject to the deployment asset check.

# Generate api mocked files

## Duckduck go
```
curl -s -X POST https://html.duckduckgo.com/html \
     --data-urlencode "q=When did people fly to the moon?" \
     --data-urlencode "b=" \
     --data-urlencode "kl=wt-wt" \
| npx html-minifier-terser \
     --collapse-whitespace \
     --remove-comments \
     --minify-css true \
     --minify-js true \
| jq -Rs '{html: .}' \
| jq -r '.html'
| pbcopy
```

### How to test the wiremock stub locally
```
curl -X POST --proxy http://localhost:7070 https://html.duckduckgo.com/html \
     -k --proxy-insecure \
     --data-urlencode "q=When did people fly to the moon?" \
     --data-urlencode "b=" \
     --data-urlencode "kl=wt-wt"
```

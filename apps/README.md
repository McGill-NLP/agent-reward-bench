# Apps

In this directory, we have the source code for the apps.

## Demo

To launch the demo, run, from the project root:

```bash
# optional: set the port to avoid conflicts
export ARB_GRADIO_PORT=21801

python apps/demo.py --port $ARB_GRADIO_PORT
```

This will launch a Gradio app that allows you to view the trajectories and annotations located inside `./agentlab_results`.

### Cloudflare

If you have cloudflare, you can deploy the app to cloudflare by running:

```bash
# optional: set the port to avoid conflicts
export ARB_GRADIO_PORT=21802

# optional: set the domain to avoid conflicts
export ARB_CF_DOMAIN="example.com"

export CF_TUNNEL=arb-demo
export CF_PORT=$ARB_GRADIO_PORT
export DOMAIN_NAME=$ARB_CF_DOMAIN

# First, if you have an existing tunnel, delete it
cloudflared tunnel cleanup $CF_TUNNEL
cloudflared tunnel delete $CF_TUNNEL

# Now, create a new tunnel and route it to the correct domain
cloudflared tunnel create $CF_TUNNEL
cloudflared tunnel route dns --overwrite-dns $CF_TUNNEL $CF_TUNNEL.$DOMAIN_NAME

# inside a screen, run:
screen -S cf-$CF_TUNNEL -dm cloudflared tunnel run  --url http://localhost:$CF_PORT $CF_TUNNEL
```
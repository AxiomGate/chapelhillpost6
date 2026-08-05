# AxiomGate Labs — Project Portfolio

A standalone, static project-portfolio site. Click a project name to see what it is and what problem it solves. No framework, no build step — plain HTML, CSS, and JavaScript, same as the rest of the AxiomGate sites.

## Local preview

```bash
cd portfolio
python3 -m http.server 8080
# open http://localhost:8080
```

## Editing content

All project data lives in one file: [`js/data.js`](js/data.js). Each project is a plain object:

```js
{
  id: 'unique-slug',
  name: 'Display Name',
  category: 'automation',       // must match an id in CATEGORIES
  status: 'active',              // live | active | prototype | concept | archived | inprogress
  summary: 'One sentence for the card.',
  whatItIs: 'A few sentences on what it actually does.',
  problem: 'A few sentences on the problem it solves.',
  tech: ['Python', 'Docker'],
  repos: [{ name: 'AxiomGate/repo-name', url: 'https://github.com/AxiomGate/repo-name' }],
}
```

Add, remove, or reorder entries in the `PROJECTS` array — the page re-renders from that array, so no HTML editing is needed. Categories and status labels are defined at the top of the same file.

## Deploying on Unraid

### Option A — Docker Compose Manager plugin (recommended)

1. Install the **Docker Compose Manager** plugin from the Unraid Community Applications store, if you don't already have it.
2. Copy this `portfolio/` folder onto your Unraid box (e.g. via the array share, `/mnt/user/appdata/axiomgate-labs-src/`, or `git clone` directly on the box).
3. In the Compose Manager UI, create a new stack pointing at this folder (it will pick up `docker-compose.yml`).
4. Bring the stack up. It builds the image from the included `Dockerfile` and starts nginx serving the site on port **8090** (host) → **80** (container).
5. Visit `http://<your-unraid-ip>:8090`.

Change the host port in `docker-compose.yml` first if 8090 is already in use on your box.

### Option B — plain `docker` commands

```bash
cd portfolio
docker build -t axiomgate-labs-portfolio .
docker run -d --name axiomgate-labs-portfolio --restart unless-stopped -p 8090:80 axiomgate-labs-portfolio
```

### Option C — Unraid's built-in Docker tab, from the image

If you'd rather not use Compose Manager: build and push the image to a registry (or `docker save`/`docker load` it onto the box), then add a container in Unraid's **Docker** tab pointing at that image, mapping container port `80` to whatever host port you like.

### Putting it behind a domain / HTTPS

This container only serves plain HTTP on the port you map. For a real domain and TLS, put it behind a reverse proxy you're already running on Unraid (Nginx Proxy Manager, SWAG, Traefik, etc.) and point a proxy host at `http://<unraid-ip>:8090`.

### Updating after editing `data.js`

Compose Manager: rebuild the stack (`docker compose up -d --build` under the hood). Plain Docker: re-run the `docker build` command above, then recreate the container — no need to touch anything else.

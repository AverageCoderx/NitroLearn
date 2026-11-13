# Vendored libraries

## three.js 0.186.1 (MIT)

`three.min.js` is the published npm package `three@0.186.1`, bundled into one classic script that defines the
global `THREE`. Nothing in it is changed; the viewer loads it from this file and never from the network.
Licence: `THREE-LICENSE.txt`.

How the file was made:

```bash
npm pack three@0.186.1        # sha256 of the tarball: 8cd068708ea44f2c73c944b1cead2ba2f0d5c15c8fc194e5700f4e4f4a033fe7
tar -xzf three-0.186.1.tgz
npx esbuild@0.25.10 package/build/three.module.js --bundle --format=iife --global-name=THREE \
    --minify --legal-comments=inline --target=es2020 --outfile=three.min.js
```

A one-line banner naming the library, version and licence is added at the top.

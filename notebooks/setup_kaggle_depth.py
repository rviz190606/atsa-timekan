"""Install official R torch CUDA binaries in Kaggle's persistent working area."""
from pathlib import Path
import concurrent.futures
import os
import subprocess
import time
import urllib.request

root = Path('/kaggle/working')
lib = root / 'R-library'
cache = root / 'depth-install-cache'
lib.mkdir(exist_ok=True)
cache.mkdir(exist_ok=True)
os.environ['R_LIBS_USER'] = str(lib)
os.environ['TMPDIR'] = '/kaggle/temp/torch-install'
Path(os.environ['TMPDIR']).mkdir(parents=True, exist_ok=True)

def r(code):
    p = subprocess.run(['Rscript', '-e', code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout, flush=True)
    p.check_returncode()

check = subprocess.run(['Rscript', '-e', 'library(torch); library(coro); stopifnot(cuda_is_available())'],
    capture_output=True, text=True)
if check.returncode == 0:
    print('PERSISTENT_R_TORCH_CUDA_READY', flush=True)
    (root/'depth_R_ready.txt').write_text('R torch CUDA is ready\n')
    raise SystemExit(0)

print('Installing missing R dependencies into', lib, flush=True)
r('''options(timeout=600,repos=c(CRAN="https://cloud.r-project.org"))
ap <- available.packages()
deps <- unique(c("coro",unlist(tools::package_dependencies(c("torch","coro"),db=ap,recursive=TRUE))))
deps <- setdiff(intersect(deps,rownames(ap)),rownames(installed.packages()))
if(length(deps)) install.packages(deps,lib=Sys.getenv("R_LIBS_USER"),Ncpus=2)
''')

url = 'https://torch-cdn.mlverse.org/packages/cu128/0.17.0/src/contrib/torch_0.17.0_R_x86_64-pc-linux-gnu.tar.gz'
size = 3831886934
archive = cache / 'torch_gpu_0.17.0.tar.gz'
chunk = 64 * 1024 * 1024
parts = [(i, i*chunk, min(size,(i+1)*chunk)-1) for i in range((size+chunk-1)//chunk)]
started = time.monotonic()

def download(part):
    i, start, end = part
    dest = cache/f'part_{i:03d}'
    needed = end-start+1
    if dest.exists() and dest.stat().st_size==needed:
        return dest
    for attempt in range(3):
        try:
            req = urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(req,timeout=90) as response:
                if response.status != 206:
                    raise RuntimeError('Server did not honor the requested byte range')
                with dest.open('wb') as f:
                    while True:
                        data = response.read(1024*1024)
                        if not data: break
                        f.write(data)
            if dest.stat().st_size != needed:
                raise RuntimeError('Incomplete download chunk')
            return dest
        except Exception:
            if attempt==2: raise

if not archive.exists() or archive.stat().st_size != size:
    with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-0'}),timeout=60) as response:
        if response.status != 206 or len(response.read()) != 1:
            raise RuntimeError('Resumable byte-range download is unavailable')
    print('Downloading official 3.83 GB torch package in resumable chunks',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        pending = [pool.submit(download,p) for p in parts]
        done_bytes = 0
        for future in concurrent.futures.as_completed(pending):
            p = future.result()
            done_bytes += p.stat().st_size
            print(f'DOWNLOAD {done_bytes/1e9:.2f}/{size/1e9:.2f} GB | {(time.monotonic()-started)/60:.1f} min',flush=True)
    with archive.open('wb') as f:
        for i, _, _ in parts:
            with (cache/f'part_{i:03d}').open('rb') as part:
                while True:
                    data=part.read(8*1024*1024)
                    if not data: break
                    f.write(data)
    if archive.stat().st_size != size: raise RuntimeError('Assembled archive has wrong size')
    # Only redundant downloaded installer chunks are removed; the archive is retained.
    for i, _, _ in parts:
        (cache/f'part_{i:03d}').unlink()

print('Installing the downloaded CUDA package',flush=True)
r(f'''install.packages("{archive}",repos=NULL,type="source",lib=Sys.getenv("R_LIBS_USER"))
library(torch); library(coro); stopifnot(cuda_is_available())
print(packageVersion("torch")); sessionInfo()
''')
(root/'depth_R_ready.txt').write_text('R torch CUDA is ready\n')
print('PERSISTENT_R_TORCH_CUDA_READY',flush=True)

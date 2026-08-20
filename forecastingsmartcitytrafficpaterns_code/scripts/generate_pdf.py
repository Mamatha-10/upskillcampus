import subprocess
import sys
from pathlib import Path

paths = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]
out = Path('reports') / 'report_generated.pdf'
out_abs = str(out.resolve())
url = f"file:///{Path.cwd().as_posix()}/reports/report_generated.html"

for p in paths:
    exe = Path(p)
    if exe.exists():
        print('Trying:', p)
        cmd = [str(exe), '--headless', '--disable-gpu', f'--print-to-pdf={out_abs}', url]
        try:
            ret = subprocess.call(cmd)
            if out.exists():
                print('Success:', p)
                sys.exit(0)
            else:
                print('Attempt failed for:', p, 'return code', ret)
        except Exception as e:
            print('Error running', p, e)

print('No browser produced a PDF')
sys.exit(2)

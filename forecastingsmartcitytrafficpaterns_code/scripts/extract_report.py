from bs4 import BeautifulSoup
from pathlib import Path

p = Path('report_executed.html')
if not p.exists():
    print('report_executed.html not found')
    raise SystemExit(1)

html = p.read_text(encoding='utf-8', errors='ignore')
soup = BeautifulSoup(html, 'html.parser')

print('Report title:', soup.title.string if soup.title else 'N/A')
print('\n--- Headings (h1-h4) and nearby text ---\n')
for h in soup.find_all(['h1','h2','h3','h4']):
    txt = ' '.join(h.get_text(strip=True).split())
    print(f'{h.name}: {txt}')
    # get next non-empty text sibling paragraphs (up to 2)
    nxt = h.find_next_siblings()
    cnt = 0
    for s in nxt:
        if s.name and s.name in ['p','div']:
            text = ' '.join(s.get_text(strip=True).split())
            if text:
                print('  ->', text[:300])
                cnt += 1
        if cnt >= 2:
            break

print('\n--- First 3 Images (src/alt) ---\n')
imgs = soup.find_all('img')
for i, img in enumerate(imgs[:3], 1):
    print(f'{i}. src={img.get("src")} alt="{img.get("alt")}"')

print('\n--- Small table previews (up to 2 tables) ---\n')
tables = soup.find_all('table')
for ti, table in enumerate(tables[:2], 1):
    rows = []
    for tr in table.find_all('tr')[:6]:
        cols = [td.get_text(strip=True) for td in tr.find_all(['td','th'])]
        rows.append(cols)
    print(f'Table {ti}:')
    for r in rows:
        print(' | '.join(r))
    print('---')

print('\n--- Looking for specific keywords and short snippets ---\n')
keywords = ['Best Model','Feature Importance','Model comparison','Actual vs Predicted','submission.csv','Predictions on Test Data','Traffic Distribution']
text = soup.get_text(separator=' ')
for kw in keywords:
    if kw in text:
        idx = text.index(kw)
        snippet = text[idx:idx+300].replace('\n',' ')
        print(f'FOUND: {kw} ->', snippet)

print('\nExtraction complete.')

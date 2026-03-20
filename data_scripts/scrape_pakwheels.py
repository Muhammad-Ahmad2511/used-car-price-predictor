import csv
import re
import time
import random

import requests
from bs4 import BeautifulSoup
import pandas as pd


REQUIRED_COLUMNS = [
    "make", "model", "year", "mileage_km", "engine_cc",
    "assembly", "reg_city", "city", "transmission", "fuel_type", "price_pkr"
]


def parse_price(text):
    try:
        text = text.lower().replace(',', '').replace('pkr', '').strip()
        m = re.search(r'(\d+\.?\d*)', text)
        if not m:
            return None
        n = float(m.group(1))
        if 'crore' in text:
            return int(n * 10_000_000)
        if 'lac' in text or 'lakh' in text:
            return int(n * 100_000)
        return int(n) if n >= 1000 else int(n * 100_000)
    except Exception:
        return None


def parse_fuel(value):
    v = value.strip().lower()
    if 'petrol'   in v: return 'Petrol'
    if 'diesel'   in v: return 'Diesel'
    if 'phev'     in v: return 'PHEV'
    if 'plug-in'  in v: return 'PHEV'
    if 'hybrid'   in v: return 'Hybrid'
    if 'electric' in v: return 'Electric'
    if 'cng'      in v: return 'CNG'
    if 'lpg'      in v: return 'LPG'
    return None


def parse_transmission(value):
    v = value.strip().lower()
    if any(x in v for x in ['automatic', 'auto', 'cvt', 'ags']): return 'Automatic'
    if 'manual' in v: return 'Manual'
    return None


def parse_cc(value):
    v = str(value).strip().lower()
    m = re.search(r'(\d[\d,]*)\s*cc', v)
    if m:
        return int(m.group(1).replace(',', ''))
    m = re.search(r'(\d+\.?\d*)\s*l(?:itre|iter)?', v)
    if m:
        litres = float(m.group(1))
        if litres < 20:
            return int(litres * 1000)
    m = re.search(r'(\d[\d,]{2,})', v)
    if m:
        val = int(m.group(1).replace(',', ''))
        if 50 <= val <= 10000:
            return val
    return None


def parse_km(value):
    v = str(value).strip().lower().replace(',', '')
    m = re.search(r'(\d+)', v)
    if m:
        val = int(m.group(1))
        if val > 0:
            return val
    return None


def clean_model(raw):
    if not raw:
        return None
    result = raw.strip()
    for pat in [r'\s+for\s+sale.*$', r'\s+in\s+[A-Z].*$',
                r'\s*\|\s*pakwheels.*$', r'\s*-\s*pakwheels.*$']:
        result = re.sub(pat, '', result, flags=re.IGNORECASE).strip()
    return result or None


def parse_engine_table(soup, record):
    table = soup.find('table', class_='table-engine-detail')
    if not table:
        return
    for td in table.find_all('td'):
        span = td.find('span', class_=re.compile(r'engine-icon'))
        if not span:
            continue
        classes = span.get('class', [])
        field = None
        for cls in classes:
            if cls != 'engine-icon':
                field = cls.lower().strip()
                break
        if not field:
            continue
        p = td.find('p')
        if not p:
            continue
        value = p.get_text(strip=True)
        if not value:
            continue
        if field == 'year' and not record['year']:
            try:
                record['year'] = int(re.sub(r'[^\d]', '', value))
            except ValueError:
                pass
        elif field in ('mileage', 'millage') and not record['mileage_km']:
            record['mileage_km'] = parse_km(value)
        elif field == 'engine' and not record['engine_cc']:
            record['engine_cc'] = parse_cc(value)
        elif field == 'transmission' and not record['transmission']:
            record['transmission'] = parse_transmission(value)
        elif field in ('fuel_type', 'type') and not record['fuel_type']:
            record['fuel_type'] = parse_fuel(value)
        elif field == 'assembly' and not record['assembly']:
            record['assembly'] = 'Imported' if 'import' in value.lower() else 'Local'


def parse_ul_featured(soup, record):
    specs_ul = soup.find('ul', class_='ul-featured')
    if not specs_ul:
        return
    items = specs_ul.find_all('li')
    i = 0
    while i < len(items) - 1:
        label = items[i].text.strip().lower()
        value = items[i + 1].text.strip()
        if 'registered' in label and not record['reg_city']:
            record['reg_city'] = value
        elif 'assembly' in label and not record['assembly']:
            record['assembly'] = 'Imported' if 'import' in value.lower() else 'Local'
        elif ('engine' in label or 'capacity' in label) and not record['engine_cc']:
            record['engine_cc'] = parse_cc(value)
        elif 'transmission' in label and not record['transmission']:
            record['transmission'] = parse_transmission(value)
        elif 'fuel' in label and not record['fuel_type']:
            record['fuel_type'] = parse_fuel(value)
        elif 'mileage' in label and not record['mileage_km']:
            record['mileage_km'] = parse_km(value)
        i += 2


def get_listing_urls(start_page, end_page, output_file):
    print(f"Collecting URLs from page {start_page} to {end_page}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    seen = {}
    base_url = "https://www.pakwheels.com/used-cars/search/-/"

    for page_num in range(start_page, end_page + 1):
        url = f"{base_url}?page={page_num}"
        print(f"  Page {page_num}/{end_page}...", end=" ", flush=True)
        try:
            response = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            links = soup.find_all('a', href=re.compile(r'/used-cars/.+-for-sale-in-.+-\d{6,}'))
            raw_page_urls = []
            for link in links:
                href = link.get('href', '')
                if href:
                    if not href.startswith('http'):
                        href = 'https://www.pakwheels.com' + href
                    raw_page_urls.append(href)
            unique_page_urls = list(dict.fromkeys(raw_page_urls))
            new_urls = [u for u in unique_page_urls if u not in seen]
            seen.update(dict.fromkeys(new_urls))
            if raw_page_urls:
                print(f"{len(new_urls)} new URLs (total: {len(seen)})")
            else:
                print("No URLs found - stopping.")
                break
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(random.uniform(2.0, 4.0))

    all_urls = list(seen.keys())
    print(f"\nTotal unique URLs to scrape: {len(all_urls)}\n")
    return all_urls


def scrape_listing(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    record = {col: None for col in REQUIRED_COLUMNS}
    title_text = ''
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        meta_desc_tag = soup.find('meta', {'name': 'description'})
        meta_desc = meta_desc_tag.get('content', '') if meta_desc_tag else ''
        title_elem = soup.find('title')
        if title_elem:
            title_text = title_elem.text.strip()
            parts = title_text.split()
            if parts:
                record['make'] = parts[0]
            year_match = re.search(r'\b(19\d{2}|20[0-2]\d)\b', title_text)
            if year_match:
                record['year'] = int(year_match.group(1))
            if record['make'] and record['year']:
                temp = title_text.replace(record['make'], '', 1)
                temp = temp.split(str(record['year']))[0]
                record['model'] = clean_model(temp)
        city_match = re.search(r'for sale in ([A-Z][a-zA-Z\s]+?)(?:\s*[-|]|$)', title_text)
        if city_match:
            record['city'] = city_match.group(1).strip()
        price_elem = soup.find('strong', class_='generic-green')
        if not price_elem:
            price_elem = soup.find('div', class_='price-box')
        if not price_elem:
            price_elem = soup.find('div', class_='sticky-ad-detail')
        if price_elem:
            record['price_pkr'] = parse_price(price_elem.get_text())
        parse_engine_table(soup, record)
        parse_ul_featured(soup, record)
        if not record['mileage_km']:
            m = re.search(r'(\d[\d,]*)\s*km', meta_desc, re.IGNORECASE)
            if m:
                record['mileage_km'] = int(m.group(1).replace(',', ''))
        if not record['transmission']:
            dl = meta_desc.lower()
            if any(x in dl for x in ['automatic', 'cvt', 'ags']):
                record['transmission'] = 'Automatic'
            elif 'manual' in dl:
                record['transmission'] = 'Manual'
        if not record['fuel_type']:
            m = re.search(r'\b(petrol|diesel|hybrid|electric|phev|cng|lpg)\b',
                          meta_desc, re.IGNORECASE)
            if m:
                record['fuel_type'] = m.group(1).title()
        if not record['reg_city'] and record['city']:
            record['reg_city'] = record['city']
        if not record['city'] and record['reg_city']:
            record['city'] = record['reg_city']
        if not record['assembly']:
            imported_makes = {'mercedes', 'bmw', 'audi', 'lexus', 'land rover',
                              'porsche', 'bentley', 'ferrari', 'volvo', 'jaguar',
                              'mini', 'infiniti', 'acura', 'genesis', 'ford'}
            make_lower = str(record['make'] or '').lower()
            record['assembly'] = (
                'Imported' if any(x in make_lower for x in imported_makes) else 'Local'
            )
    except Exception:
        pass
    return record


def scrape_all(urls, output_file):
    print(f"Scraping {len(urls)} unique URLs...\n")
    csv_file = open(output_file, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_file, fieldnames=REQUIRED_COLUMNS)
    writer.writeheader()
    success = skipped = 0
    for i, url in enumerate(urls, 1):
        record = scrape_listing(url)
        if record['price_pkr'] and record['make']:
            writer.writerow(record)
            csv_file.flush()
            success += 1
            print(f"[{i}/{len(urls)}] {record.get('make')} {record.get('model')} | "
                  f"{record.get('city')} | {record.get('fuel_type')} | "
                  f"Rs {record.get('price_pkr'):,}")
        else:
            skipped += 1
            print(f"[{i}/{len(urls)}] Skipped")
        time.sleep(random.uniform(2.0, 4.0))
    csv_file.close()
    print(f"\nDone. {success} rows saved, {skipped} skipped.")

def main():
    start = int(input("Enter start page: "))
    end = int(input("Enter end page: "))
    output_file = "pakwheels_scraped.csv"

    if start > end:
        print(f"Error: start ({start}) cannot be greater than end ({end})")
        return

    print(f"\nPakWheels Scraper")
    print(f"Pages: {start} to {end} | Output: {output_file}\n")

    urls = get_listing_urls(start, end, output_file)
    if not urls:
        print("No URLs collected.")
        return

    scrape_all(urls, output_file)
   
if __name__ == "__main__":
    main()

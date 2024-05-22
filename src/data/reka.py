import requests
import csv
import xml.etree.ElementTree as ET
from unidecode import unidecode

def fetch_and_append_data(api_url, target_reka, csv_file):
    response = requests.get(api_url)
    xml_data = response.content

    root = ET.fromstring(xml_data)

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        file.seek(0, 2)
        if file.tell() == 0:
            writer.writerow([
                'Postaja Sifra', 'Ge Dolzina', 'Ge Sirina', 'Kota 0', 'Reka', 'Merilno Mesto', 
                'Ime Kratko', 'Datum', 'Datum CET', 'Vodostaj', 'Pretok', 'Pretok Znacilni', 
                'Temp Vode', 'Prvi VV Pretok', 'Drugi VV Pretok', 'Tretji VV Pretok'
            ])

        for postaja in root.findall('postaja'):
            reka = postaja.find('reka').text


            if reka == target_reka:
                postaja_sifra = postaja.get('sifra')
                ge_dolzina = postaja.get('ge_dolzina')
                ge_sirina = postaja.get('ge_sirina')
                kota_0 = postaja.get('kota_0')
                merilno_mesto = unidecode(postaja.find('merilno_mesto').text)
                ime_kratko = unidecode(postaja.find('ime_kratko').text)
                datum = postaja.find('datum').text
                datum_cet = postaja.find('datum_cet').text
                vodostaj = postaja.find('vodostaj').text
                pretok = postaja.find('pretok').text if postaja.find('pretok') is not None else ''
                pretok_znacilni = postaja.find('pretok_znacilni').text if postaja.find('pretok_znacilni') is not None else ''
                temp_vode = postaja.find('temp_vode').text
                prvi_vv_pretok = postaja.find('prvi_vv_pretok').text if postaja.find('prvi_vv_pretok') is not None else ''
                drugi_vv_pretok = postaja.find('drugi_vv_pretok').text if postaja.find('drugi_vv_pretok') is not None else ''
                tretji_vv_pretok = postaja.find('tretji_vv_pretok').text if postaja.find('tretji_vv_pretok') is not None else ''

                writer.writerow([
                    postaja_sifra, ge_dolzina, ge_sirina, kota_0, reka, merilno_mesto, 
                    ime_kratko, datum, datum_cet, vodostaj, pretok, pretok_znacilni, 
                    temp_vode, prvi_vv_pretok, drugi_vv_pretok, tretji_vv_pretok
                ])

    print(f"Data for '{target_reka}' has been appended to {csv_file}")


api_url = 'http://www.arso.gov.si/xml/vode/hidro_podatki_zadnji.xml'
target_reka = 'Drava' 
csv_file = 'data/raw/reka_data.csv'

fetch_and_append_data(api_url, target_reka, csv_file)

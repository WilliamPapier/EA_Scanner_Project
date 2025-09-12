import requests

# Test trade setup for SimpleML_EA
setup = [{
    'instrument': 'US30m',
    'entry_price': 35000,
    'sl_price': 34900,
    'tp_price': 35100,
    'lot_size': 0.1,
    'is_buy': True,
    'confidence': 0.95,
    'session': 'London'
}]

print('Sending request to backend...')
try:
    resp = requests.post('http://127.0.0.1:5000/scanner', json=setup)
    print('Response status:', resp.status_code)
    print('Response text:', resp.text)
except Exception as e:
    print('Exception occurred while sending request:', e)

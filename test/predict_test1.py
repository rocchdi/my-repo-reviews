import os
import requests
import json


api_address='my_api_container'
# port de l'API
api_port = 8000

# test is API repond


r = requests.get(
        url='http://{address}:{port}/'.format(address=api_address, port=api_port) 
)

#print('status_code=', r.status_code)

output = '''
============================
    verifier si api fonctionne 
============================

request done at "/"

expected result = 200
actual result = {status_code}

==>  {test_status}

'''




# statut de la requête
status_code = r.status_code

# affichage des résultats
if status_code != 200:
    test_status = 'FAILURE'
else:
    test_status = 'SUCCESS'


output=output.format(status_code=status_code, test_status=test_status)
print(output)


if (test_status == 'SUCCESS'): #api fonctionne
    # on lance les tests

    #-------------------------------------
    # requête1
    #------------------------------------

    r = requests.get(
        url='http://{address}:{port}/predict'.format(address=api_address, port=api_port),
        params= {
            'model': 'BOW_Regression',
            'sentence': 'hello disney'
        },
        headers={'authorization-header': 'Basic alice:wonderland'}
    )

    output = '''
============================
predict  test1: request complet de l 'api sur le modele BOW_Regression 
============================

request done at "/predict"
| model: BOW_Regression
| sentence: hello disney
| authorization-header: Basic alice:wonderland

expected result = 200
actual result = {status_code}

==>  {test_status}

    '''


    # statut de la requête
    status_code = r.status_code

    # affichage des résultats
    if status_code == 200:
        test_status = 'SUCCESS'
    else:
        test_status = 'FAILURE'

    output=output.format(status_code=status_code, test_status=test_status)
    print(output)



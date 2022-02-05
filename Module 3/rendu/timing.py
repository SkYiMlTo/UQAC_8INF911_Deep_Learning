import time
from datetime import datetime


def timing(func):
    """
    Mesure le temps d'exécution d'une fonction.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print("[" + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "] - Début execution de la fonction " + str(func).split('.')[1])
        res = func(*args, **kwargs)
        end_time = time.time()
        print("[" + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "] - Fin execution de la fonction " +
              str(func).split('.')[1])
        print("Durée d'exécution : " + str(round(end_time - start_time, 2)) + "s")
        return res

    return wrapper

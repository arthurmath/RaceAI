import multiprocessing as mp
import numpy as np
import time
import os

"""MAP : applique la fonction à tous les éléments de la séquence et attend que toutes les tâches soient terminées 
avant de retourner les résultats sous forme d'une liste. C'est une opération synchrone, ce qui signifie 
que le programme reste bloqué jusqu'à ce que tous les résultats soient disponibles. Lourd en mémoire si grande liste"""

"""IMAP : similaire à map, mais retourne un itérateur au lieu d'une liste complète. Les résultats sont produits 
 et peuvent être consommés dès qu'ils sont disponibles, sans attendre que toutes les tâches soient terminées.
Asynchronisme : imap est plus efficace pour traiter de grandes séquences, car il permet de commencer à travailler
 sur les résultats dès qu'ils sont prêts, plutôt que de devoir attendre que toute la séquence soit traitée."""


# print("Number of CPU:", mp.cpu_count())




# ## MAP
# """La méthode découpe l'itérable en un nombre de morceaux qu'elle envoie au pool de processus 
#  comme des tâches séparées. Peut entraîner une grosse consommation de mémoire pour les itérables très longs."""
 
# def f(x):
#     return x*x

# if __name__ == '__main__':
#     with mp.Pool(processes=5) as pool:
#         print(pool.map(f, [1, 2, 3]))

# # >>> [1, 4, 9]




# ## PROCESSUS
# """Un processus est une instance indépendante d'un programme en cours d'exécution, 
# avec son propre espace mémoire. Les processus sont isolés les uns des autres."""
# def info(title):
#     print(title)
#     print('module name:', __name__)
#     print('parent process:', os.getppid())
#     print('process id:', os.getpid())
#     print("")

# def func(name):
#     info('function f')
#     print('hello', name)

# if __name__ == '__main__':
#     info('main line')
#     p = mp.Process(target=func, args=('bob',))
#     p.start()
#     p.join(timeout=None)   
#     # Le processus parent attend que le processus enfant finisse son exécution avant de continuer.
#     # Si timeout est un nombre positif, elle bloque au maximum pendant timeout secondes.




### POOL 
""" Un pool est un groupe de processus travailleurs (workers) 
qui peuvent exécuter des tâches en parallèle. """
def f(x):
    return x*x

if __name__ == '__main__':
    # start 4 worker processes
    with mp.Pool(processes=4) as pool:

        # print "[0, 1, 4,..., 81]"
        print(pool.map(f, range(10)))

        # print same numbers in arbitrary order
        for i in pool.imap_unordered(f, range(10)):
            print(i)

        # evaluate "f(20)" asynchronously
        res = pool.apply_async(f, (20,))        # runs in *only* one process
        print(res.get(timeout=1))               # prints "400"

        # launching multiple evaluations asynchronously, *may* use more processes
        multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
        print([res.get(timeout=1) for res in multiple_results]) # prints the PID of that process

        # make a single worker sleep for 10 seconds
        res = pool.apply_async(time.sleep, (10,))
        try:
            print(res.get(timeout=1))
        except mp.TimeoutError:
            print("We lacked patience and got a multiprocessing.TimeoutError")

        print("For the moment, the pool remains available for more work")

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")





### VQA Strut. meca
# """ Runs the optimization algorithm ntests times """
# opt_res = []
# ntests = 10
# paral = True
# items = range(1, ntests + 1)
# try:
#     if paral:
#         # Creates a process pool that uses all cpus
#         with mp.Pool() as pool:
#             # Calls the function for each item in parallel
#             for res in pool.imap_unordered(optimize, items):
#                 opt_res.append(res)
#     else:
#         opt_res.extend(optimize(i) for i in items)

# # In case we want to stop during one of many tests
# except KeyboardInterrupt:
#     if paral:
#         # We have to stop the children
#         if mp.parent_process() is not None:
#             exit(107)








import multiprocessing as mp

def process_pilot(args):
    """
    Fonction à exécuter dans un processus séparé pour chaque pilote.
    Elle crée une session, exécute la simulation, et renvoie l'index du pilote,
    sa fitness et sa progression.
    
    Args:
        args (tuple): (idx, pilot, display)
    
    Returns:
        tuple: (idx, fitness, progression)
    """
    idx, pilot, display = args
    # Création et exécution de la session
    ses = Session(train=True, player=2, agent=pilot, display=display)
    ses.run()
    
    # Calcul de la fitness et récupération de la progression
    fitness = pilot.compute_fitness(ses.car)
    progression = ses.car.progression
    
    return idx, fitness, progression

if __name__ == "__main__":
    # Préparez la liste des arguments à passer à chaque processus.
    # Par exemple, ici on garde display=True pour tous, à ajuster si nécessaire.
    tasks = [(idx, pilot, True) for idx, pilot in enumerate(self.pilots)]
    
    # Utilisation d'un pool de processus
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Lancer de manière asynchrone avec map_async
        async_result = pool.map_async(process_pilot, tasks)
        
        # Attendre la fin de tous les traitements
        async_result.wait()
        results = async_result.get()
    
    # Trier les résultats par index (optionnel, si l'ordre est important)
    results.sort(key=lambda x: x[0])
    
    # Récupération des résultats et affichage
    for idx, fitness, progression in results:
        self.fitness.append(fitness)
        self.scores.append(progression)
        print(f"Pilot {idx+1}/{self.nbPilotes}, fitness: {fitness:.3f}, progression: {progression:.3f}%")

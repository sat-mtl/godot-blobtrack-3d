# godot-3d-blobtrack

3D gpu blob tracking for godot.


## TODO:
list de choses qui pourraient être améliorées

- faire que les kalman trackers ne crééent pas un nouveau blob à chaque prédiction
  - en général, y'a beaucoup d'allocation mémoire dans la partie CPU, faudrait allouer statiquement
    - ça implique ajouter des limites dure pour les mémoires de blobs et les trackers potentiellement
- utiliser du temps en secondes float plutot que des numéro de frames pour les filtres de kalman, le max_age et la durée des mémoires
  - ça fait que le comportement du tracking dépend du framerate
  - il faut comprendre les filtres de kalman assez pour introduire la variable temps dans predict/update
- rouler l'id tracking sur GPU
  - tout ce qui est kalman + filtrage peut se faire quand même bien
  - tout ce qui est de matcher les nouveaux blobs avec les anciens ID c'est un problème de d'assignation de coût/traveling salesman. Je pense qu'un algo greedy parallèle sur GPU pourrait obtenir de bons résultats de manière non déterministe sans garantir d'optimalité mais y faut tester.
- changer le format d'output du blobtrack pour pouvoir lire tout d'un seul buffer avec un seul call de `buffer_get_data_async`
- Voir si les valeurs de dispatch `grid_table`, `grid_cluster` et `scan_blocks` peuvent être dynamiques plutôt que calculées sur `max_points`

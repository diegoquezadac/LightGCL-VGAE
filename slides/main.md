---
marp: true
theme: default
paginate: true
footer: 6 de Noviembre, 2024
math: katex
---

<style>
    
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}

.center-align {
    margin-right: auto !important;
    margin-left: auto !important;
    text-align: center;
}

.right-align {
    text-align: right;
}

.figure-container {
  display: flex;
  justify-content: space-evenly;
  margin: 0 0 0 0;
  padding: 0 0 0 0;
}

</style>

# Sobre la Utilidad de VGAE en LightGCL


#
Matías Francia
Diego Quezada

---
# Contexto

* GNNs han demostrado su efectividad en sistemas recomendadores basados en grafos.
* La mayoría de los modelos de filtrado colaborativo basados en GNNs aprenden de manera supervisada, necesitando datos etiquetados.
* En la práctica aprender representaciones de usuarios e ítems es un gran desafío debido a que la matriz de interacción es sparse.
* GNNs integradas con contrastive learning han demostrado un desempeño superior en la tarea de recomendación junto a su esquema de data augmentation.


---
# Problema

Aprender representaciones de usuarios e ítems a partir de matrices de interacción sparse, capturando tanto información local como global del grafo.


---
# LightGCL

Explicación simple

---
# Contribución

* Evaluar la utilidad de VGAE para reconstruir la matriz de interacción en el framework LightGCL.
* **Inspiración** VGAE codifica la información colaborativa en el espacio latente pues este debe cumplir con la hipótesis de clustering. Esto nos debería llevar a una mejorar reconstrucción en comparación a SVD.

<!--* ¿Por qué? La reconstrucción de la matriz de interacción realizada por SVD no considera relaciones a nivel de nodo. Qué pasa con la información colaborativa entre nodos ? co-ocurrencia ? -->


---
# Conjunto de datos Yelp

<div style="display: flex; justify-content: space-around;">

  <!-- First Table with Bottom Caption -->
  <div style="text-align: center;">
    <table style="font-size: 20px; margin: 10px; text-align: center;">
      <tr>
        <th>Métrica</th>
        <th>Valor</th>
      </tr>
      <tr>
        <td>Número de usuarios</td>
        <td>29,601</td>
      </tr>
      <tr>
        <td>Número de ítems</td>
        <td>24,734</td>
      </tr>
      <tr>
        <td>Número de interacciones</td>
        <td>1,069,128</td>
      </tr>
      <tr>
        <td>Promedio de interacciones por usuario</td>
        <td>36.12</td>
      </tr>
      <tr>
        <td>Promedio de interacciones por ítem</td>
        <td>43.23</td>
      </tr>
      <tr>
        <td>Máximo de interacciones por usuario</td>
        <td>1,282</td>
      </tr>
      <tr>
        <td>Máximo de interacciones por ítem</td>
        <td>926</td>
      </tr>
      <tr>
        <td>Densidad de la matriz</td>
        <td>0.15%</td>
      </tr>
    </table>
    <div style="font-weight: bold; font-size: 20px; margin-top: 5px;">Entrenamiento</div>
  </div>

  <!-- Second Table with Bottom Caption -->
  <div style="text-align: center;">
    <table style="font-size: 20px; margin: 10px; text-align: center;">
      <tr>
        <th>Métrica</th>
        <th>Valor</th>
      </tr>
      <tr>
        <td>Número de usuarios</td>
        <td>29,530</td>
      </tr>
      <tr>
        <td>Número de ítems</td>
        <td>24,679</td>
      </tr>
      <tr>
        <td>Número de interacciones</td>
        <td>305,466</td>
      </tr>
      <tr>
        <td>Promedio de interacciones por usuario</td>
        <td>10.32</td>
      </tr>
      <tr>
        <td>Promedio de interacciones por ítem</td>
        <td>12.35</td>
      </tr>
      <tr>
        <td>Máximo de interacciones por usuario</td>
        <td>370</td>
      </tr>
      <tr>
        <td>Máximo de interacciones por ítem</td>
        <td>269</td>
      </tr>
      <tr>
        <td>Densidad de la matriz</td>
        <td>0.04%</td>
      </tr>
    </table>
    <div style="font-weight: bold; font-size: 20px; margin-top: 5px;">Prueba</div>
  </div>

</div>


---

<div style="display: flex; justify-content: space-around; margin-top: 20px;">

  <img src="../images/interactions_per_user_train.png" alt="Interacciones por usuario (Entrenamiento)" style="width: 50%; height: auto;"/>
  
  <img src="../images/interactions_per_user_test.png" alt="Interacciones por usuario (Prueba)" style="width: 50%; height: auto;"/>

</div>

---

<div style="display: flex; justify-content: space-around; margin-top: 20px;">

  <img src="../images/interactions_per_item_train.png" alt="Interacciones por item (Entrenamiento)" style="width: 50%; height: auto;"/>
  
  <img src="../images/interactions_per_item_test.png" alt="Interacciones por item (Prueba)" style="width: 50%; height: auto;"/>

</div>

---
# Implementación


---
## LightGCL

```python
class LightGCL(nn.Module):
    def __init__(self, n_u, n_i, u_mul_s,
        v_mul_s,
        ut, 
        vt, 
        train_csr,
        adj_norm, 
        l,   
        temp,
        lambda_1,
        lambda_2,
        dropout,
        batch_user,
        device):

    pass
```

---

# Referencias

Agregar LightGCL

Agregar Variational Graph Auto-Encoders
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

# What’s Behind the Mask: Understanding Masked Graph Modeling for Graph Autoencoders


#
Matías Francia

Diego Quezada

---
# Introducción

<!-- Abstract, introduction -->

- Éxito de graph SSL en varias áreas, particularmente en química y ciencas biomédicas, donde anotar datos es costoso

- Contrastive learning: funciona bien, pero depende demasiado en las tareas de creación de vistas aumentadas (*domain-specific*)

- Generative learning (GAE): no depende de la técnica de aumentación de datos, pero sobre-enfatiza la información próxima, a costa de la estructural

---
# Trabajo relacionado

<div style="display: flex; justify-content: space-around; margin-top: 20px;">
  
  <img src="images/masking_methods.png" alt="Interacciones por usuario (Prueba)" style="width: 80%; height: auto;"/>

</div>

<!-- Hay enfoques contemporáneos que también tratan de aprovechar las virtudes del enmascaramiento en grafos, pero carecen de respaldo teórico. Sólo muestran resultados empíricos -->

---
# Formulación del problema

- Sea $G = (V, E)$ un grafo no dirigido y no ponderado
- $V = \{ v_i \}$: conjunto de nodos
- $E \subseteq V \times V$: conjunto de aristas
- $\forall v \in V, \exist x_v \in \mathbb{R}^d$: vector de características de cada nodo
- $Z = \{ z_i \}_{i=1}^{|V|}$: representación latente

- **Tarea**: Aprender *encoder* de grafo $f_\theta$ que mapea el grafo $G$ a sus representaciones latentes de baja dimensión


---

# MaskGAE

---

### Estrategia de Masking


---

### Encoder - Decoder


---

### Función de pérdida

---

# Experimentos

**Tareas**:
1. Predicción de links
2. Clasificación de nodos

**Datasets**:
- Cora, CiteSeer, Pubmed, Photo, Computer, arXiv, MAG, Collab

**Evaluación**: 

- Links: Predicción sobre muestreo de 10% de arcos que existen y 10% que no
- Nodos: Linear probing sobre las representaciones agregadas de cada capa (sigmoide)


---

# Conclusión


---

# Referencias

[1] Li, J., Wu, R., Sun, W., Chen, L., Tian, S., Zhu, L., ... & Wang, W. (2023, August). What's Behind the Mask: Understanding Masked Graph Modeling for Graph Autoencoders. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 1268-1279).
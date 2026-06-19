

# Prioridades

1. Repensar quantidades de execução por mapa
   - depende do tamanho do mapa?

1. Selecionar ambientes/mapas que serão usados

1. Usar os atributos nos gráficos -> agrupar por 1 atributo ou por 2 atributos (gráfico de barra dividido)


# Alvo

- Fazer comparações entre os modelos visões locais e globais, diferentes formas de representar, e variando histórico (1 e 5)
- Em vários ambientes, sem necessidade de pegar chave, etc
- Reportar os resultados entre modelos
  * openai parece ir especialmente mal na visão local


# Outros

1. Pedir para deixar de identificar unicamente por meio de nome e trocar por um conjunto de atributos como identificação???

1. No código do Streamlit
   * rever sumário
   * ao selecionar um item superior, restar os inferiores
   * conferir se está funcionando com novo sistema de atributos nos payloads

1. Fazer plots intra-configuração (de um experimento)
   * variabilidade dos resultados entre runs

1. Selecionar seeds para cada ambiente
   * conferir qual seed gera qual observação inicial
   * testar vários


# Passado

- Testes com tamanhos de históricos variados 1, 3, 5 e 9
   * 3 e 5 com diferentes vantagens em diferentes situações
   * 3 - pequeno, deixa mais dependente do modelo
   * 5 - testa se o modelo é capaz de aprender com histórico maior de decisões anteriores e se adaptar
   * 9 não ajudou no desempenho - caiu ou manteve
   * vou usar 5 - são as duas últimas interações (observações + resposta do modelo), acrescida da nova observação
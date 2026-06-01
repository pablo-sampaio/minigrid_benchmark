

# Prioridades

1. (Del)Testar as modificações (rodar note de openai). Depois partir para o item abaixo. (Acho melhor pular para o item abaixo)

1. Criar mecanismos de setar atributos por "experimento" ou por "run"?
   * pedir para deixar de identificar unicamente por meio de nome e trocar por um conjunto de atributos como identificação
   * fazer alterações nos antigos?? (não vale a pena, por conta de diferenças nos prompts)

1. Consertar/ajustar e organizar prompts.
   * criar arquivo separado?? 
   * apenas rever nomes das variáveis?? 
   * ou criar função com parâmetros (global, separador, numerar, incluir_exemplo)??
   * prompts locais: informar que a observação informa a direção global do player

1. Salvar a pasta de resultados antiga separada, com note de plot. Criar uma pasta vazia. Só fazer commit quando: 
   - refactor para usar atributos de execução (no lugar do nome)
   - testar execução do experimento
   - testar o plot 

1. Usar os atributos nos gráficos -> agrupar por 1 atributo ou por 2 atributos (gráfico de barra dividido)

1. Corrigir o dashboard de streamlit para lidar com o novo sistema de atributos


# Alvo

- Fazer comparações entre os modelos com formato 1 (simples) e 2 (com separadores e números), locais e globais
- Em vários ambientes, sem necessidade de pegar chave, etc
- Reportar os resultados entre modelos
  * openai parece ir especialmente mal


# Outros

1. No código do Streamlit
   * rever sumário
   * ao selecionar um item superior, restar os inferiores

1. Fazer plots intra-configuração (de um experimento)
   * variabilidade dos resultados entre runs

1. Selecionar seeds para cada ambiente
   * conferir qual seed gera qual observação inicial
   * testar vários

1. Escolher outros ambientes
   * todos de Lava e outros de ações similares aos de Lava

1. Fazer benchmark executar com variações de prompt e históricos - vai receber só o modelo!
   * deixar todos os prompts em inglês!
   * depende de definir: tamanho de histórico, ambientes selecionados


# Decisões


1. Testes com tamanhos de históricos variados 3, 5 e 9
   * 3 e 5 com diferentes vantagens em diferentes situações
   * 3 - pequeno, deixa mais dependente do modelo
   * 5 - testa se o modelo é capaz de aprender com histórico maior de decisões anteriores e se adaptar
   * 9 não ajudou no desempenho - caiu ou manteve
   * vou usar 5 - duas últimas interações (observações + resposta do modelo), acrescida da nova observação
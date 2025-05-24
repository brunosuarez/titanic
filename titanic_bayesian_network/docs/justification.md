# Descrição das Variáveis - Titanic

## Variável Alvo
- **Survived**: Indica se o passageiro sobreviveu ao desastre
  - Valores: 
    - `0` = Não sobreviveu
    - `1` = Sobreviveu

## Variáveis Preditivas
- **Pclass**: Classe do ticket no navio
  - Valores:
    - `1` = 1ª classe (mais cara)
    - `2` = 2ª classe
    - `3` = 3ª classe (mais barata)

- **Sex**: Gênero do passageiro
  - Valores:
    - `male` = Masculino
    - `female` = Feminino

- **Age**: Idade do passageiro (discretizada)
  - Categorias:
    - `Child` = Menor que 12 anos
    - `Teen` = 12-18 anos
    - `Adult` = 19-60 anos
    - `Elderly` = Acima de 60 anos

- **Fare**: Preço pago pelo ticket (discretizado)
  - Categorias:
    - `Low` = Tarifa baixa
    - `Medium` = Tarifa média
    - `High` = Tarifa alta

- **SibSp**: Número de irmãos/cônjuges a bordo
  - Valores: Inteiro de 0 a 8

- **Parch**: Número de pais/filhos a bordo
  - Valores: Inteiro de 0 a 6
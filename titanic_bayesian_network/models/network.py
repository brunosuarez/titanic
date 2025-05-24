import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import os

# Função para pré-processar os dados
def preprocess_data(data):
    # Categorizar a variável Age em faixas etárias
    data['AgeGroup'] = pd.cut(
        data['Age'],
        bins=[0, 12, 18, 60, 120],
        labels=['Child', 'Teen', 'Adult', 'Senior']
    )

    # Categorizar a variável Fare em faixas de preço
    data['FareGroup'] = pd.cut(
        data['Fare'],
        bins=[-1, 7, 15, 1000],
        labels=['Low', 'Medium', 'High']
    )

    # As variáveis Sex, Pclass e Survived já são categóricas, então vamos garantir que estejam no formato correto
    data['Sex'] = data['Sex'].astype('category')
    data['Pclass'] = data['Pclass'].astype('category')
    data['Survived'] = data['Survived'].astype('category')

    return data

# Função para criar uma CPD (Tabela de Probabilidade Condicional)
def create_cpd(data, variable):
    counts = data[variable].value_counts(normalize=True).sort_index()
    values = [[p] for p in counts.values]  # transforma shape para (n,1)
    cpd = TabularCPD(
        variable=variable,
        variable_card=len(counts),
        values=values
    )
    return cpd

# Função para construir a rede bayesiana
def build_network(data):
    print("🔧 Construindo rede bayesiana...")

    # Definição da estrutura da rede (arcos)
    model = DiscreteBayesianNetwork([
        ('Pclass', 'Survived'),
        ('Sex', 'Survived'),
        ('AgeGroup', 'Survived'),
        ('FareGroup', 'Survived')
    ])

    # Criar CPDs para as variáveis sem pais
    cpd_pclass = create_cpd(data, 'Pclass')
    cpd_sex = create_cpd(data, 'Sex')
    cpd_agegroup = create_cpd(data, 'AgeGroup')
    cpd_faregroup = create_cpd(data, 'FareGroup')

    # Criar CPD para Survived condicionada nos pais (Pclass, Sex, AgeGroup, FareGroup)
    evidence_vars = ['Pclass', 'Sex', 'AgeGroup', 'FareGroup']
    group = data.groupby(evidence_vars + ['Survived']).size().unstack(fill_value=0)

    # Ordenar índices para manter a mesma ordem dos estados
    group = group.reindex(index=pd.MultiIndex.from_product(
        [data[v].cat.categories for v in evidence_vars]
    ), fill_value=0)

    # Normalizar para probabilidade condicional P(Survived | evidências)
    prob_survived = group.div(group.sum(axis=1), axis=0).fillna(0)

    # Verificando o formato da tabela de probabilidade
    print(prob_survived.head())

    # Alteração para garantir que a tabela de valores tenha 2 linhas e o número correto de colunas
    values = prob_survived.values.T.tolist()  # Transposta e transformada em lista

    # Verificar e corrigir as probabilidades para cada combinação de evidência
    for i in range(len(values[0])):
        prob_0 = values[0][i]
        prob_1 = values[1][i]

        # Verificar se estamos tentando dividir por zero
        if abs(prob_0 + prob_1 - 1) > 1e-5:  # Caso a soma das probabilidades não seja 1
            print(f"Atenção: As probabilidades não somam 1 para a combinação {i}. Prob_0: {prob_0}, Prob_1: {prob_1}")
            
            # Se prob_0 e prob_1 são ambos 0, atribuímos valores padrão (0.5, 0.5)
            if prob_0 == 0 and prob_1 == 0:
                values[0][i] = 0.5
                values[1][i] = 0.5
            else:
                # Caso a soma seja diferente de 1, normalizamos
                total = prob_0 + prob_1
                values[0][i] = prob_0 / total  # Normalizar para somar 1
                values[1][i] = prob_1 / total  # Normalizar para somar 1

    # Atualizar o CPD com as correções
    cpd_survived = TabularCPD(
        variable='Survived',
        variable_card=2,  # Apenas 2 possíveis valores para Survived (0 ou 1)
        values=values,
        evidence=evidence_vars,
        evidence_card=[len(data[v].cat.categories) for v in evidence_vars]
    )

    # Adicionar CPDs ao modelo
    model.add_cpds(cpd_pclass, cpd_sex, cpd_agegroup, cpd_faregroup, cpd_survived)

    # Validar modelo
    model.check_model()

    print("✅ Rede bayesiana criada com sucesso!")
    return model

# Função principal
def main():
    print("🚀 Processando dados do Titanic...")
    
    # Carregar o conjunto de dados Titanic
    data_path = os.path.join('data', 'train.csv')  # Caminho correto para o arquivo
    try:
        df = pd.read_csv(data_path)  # Ler o arquivo train.csv
    except FileNotFoundError:
        print(f"Erro: O arquivo '{data_path}' não foi encontrado no diretório.")
        return

    df = preprocess_data(df)  # Processar dados

    # Construir a rede bayesiana
    model = build_network(df)

    # Inferência simples
    infer = VariableElimination(model)

    # Mapeamento de valores categóricos de FareGroup, Sex, AgeGroup e Pclass para valores numéricos
    fare_group_map = {'Low': 0, 'Medium': 1, 'High': 2}
    sex_map = {'male': 0, 'female': 1}  # Mapeamento para 'Sex'
    age_group_map = {'Child': 0, 'Teen': 1, 'Adult': 2, 'Senior': 3}  # Mapeamento para 'AgeGroup'
    pclass_map = {1: 0, 2: 1, 3: 2}  # Mapeamento para 'Pclass'

    # Exemplo: probabilidade de sobrevivência dado Pclass=1, Sex=female, AgeGroup=Adult, FareGroup=High
    q = infer.query(variables=['Survived'], evidence={
        'Pclass': pclass_map[1],  # Usando o mapeamento numérico para 'Pclass'
        'Sex': sex_map['female'],  # Usando o mapeamento numérico para 'Sex'
        'AgeGroup': age_group_map['Adult'],  # Usando o mapeamento numérico para 'AgeGroup'
        'FareGroup': fare_group_map['High']  # Usando o mapeamento numérico para 'FareGroup'
    })

    print("\nProbabilidade de sobrevivência dado Pclass=1, Sex=female, AgeGroup=Adult, FareGroup=High:")
    print(q)

    # Testar outra consulta: Probabilidade de sobrevivência dado Pclass=3 e Sex=male
    q2 = infer.query(variables=['Survived'], evidence={
        'Pclass': pclass_map[3],  # Usando o mapeamento numérico para 'Pclass'
        'Sex': sex_map['male']  # Usando o mapeamento numérico para 'Sex'
    })

    print("\nProbabilidade de sobrevivência dado Pclass=3, Sex=male:")
    print(q2)

if __name__ == '__main__':
    main()

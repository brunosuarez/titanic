import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. CARREGAR OS DADOS
# Certifique-se de que 'train.csv' está na mesma pasta do seu script Python
data = pd.read_csv('train.csv')

# 2. PRÉ-PROCESSAMENTO (exemplo simplificado)
# Discretizar idade (criança/adulto/idoso)
data['AgeGroup'] = pd.cut(
    data['Age'], 
    bins=[0, 12, 60, 100], 
    labels=['Child', 'Adult', 'Elderly']
)

# Discretizar tarifa (baixa/média/alta)
data['FareGroup'] = pd.cut(
    data['Fare'], 
    bins=[0, 50, 100, 600], 
    labels=['Low', 'Medium', 'High']
)

# Tratar valores faltantes (simplificado)
data['AgeGroup'].fillna('Adult', inplace=True)  # Preenche missing com 'Adult'
data['FareGroup'].fillna('Medium', inplace=True)

# 3. DEFINIR A REDE BAYESIANA (exemplo)
model = BayesianNetwork([
    ('Pclass', 'Survived'),
    ('Sex', 'Survived'),
    ('AgeGroup', 'Survived'),
    ('FareGroup', 'Survived')
])

# 4. CRIAR AS CPTs BASEADAS NOS DADOS REAIS (exemplo para Sex -> Survived)
# Contar quantos homens/mulheres sobreviveram
survived_sex = data.groupby(['Sex', 'Survived']).size().unstack()
# Calcular probabilidades condicionais P(Survived | Sex)
survived_sex = survived_sex.div(survived_sex.sum(axis=1), axis=0)

# Criar a CPT para Sex -> Survived
cpd_sex = TabularCPD(
    variable='Survived',
    variable_card=2,  # 0 = Não, 1 = Sim
    values=[survived_sex[0].values, survived_sex[1].values],  # Valores reais!
    evidence=['Sex'],
    evidence_card=[2]
)

# Adicionar CPT ao modelo
model.add_cpds(cpd_sex)

# 5. FAZER INFERÊNCIA (exemplo)
infer = VariableElimination(model)
result = infer.query(variables=['Survived'], evidence={'Sex': 'female'})
print(result)
# -*- coding: utf-8 -*-
from fpdf import FPDF, XPos, YPos

OUT = "guilherme_ramos_aula15Teorico.pdf"

class PDF(FPDF):
    def header(self):
        pass

    def section(self, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(220, 220, 220)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_font("Helvetica", "", 11)
        self.ln(1)

    def subsection(self, title):
        self.set_font("Helvetica", "BI", 11)
        self.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 11)

    def body(self, text):
        self.multi_cell(0, 6, text)
        self.ln(1)

    def row(self, cols, widths, bold=False):
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 10)
        for col, w in zip(cols, widths):
            self.cell(w, 7, col, border=1)
        self.ln()
        self.set_font("Helvetica", "", 10)

pdf = PDF()
pdf.set_margins(20, 20, 20)
pdf.add_page()

# Titulo
pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 10, "Aula 15 - Exercicio Teorico: Aprendizado Supervisionado (Parte 2)",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
pdf.set_font("Helvetica", "", 11)
pdf.cell(0, 7, "Guilherme Ramos",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
pdf.ln(4)

# ── QUESTAO 1 ──────────────────────────────────────────────────────────────────
pdf.section("Questao 1 (0,5 pt) - Perguntas sobre SVM")

pdf.subsection("O que sao vetores de suporte em um SVM?")
pdf.body(
    "Vetores de suporte sao as amostras de treinamento que ficam mais proximas do hiperplano "
    "de separacao, ou seja, as que estao na fronteira da margem. Sao os unicos pontos que "
    "influenciam diretamente a posicao e orientacao do hiperplano. Remover qualquer outra "
    "amostra do conjunto de treino nao alteraria o modelo final."
)

pdf.subsection("O que representa o hiperplano em um SVM?")
pdf.body(
    "O hiperplano e a fronteira de decisao que separa as classes no espaco de caracteristicas. "
    "Em um espaco de n dimensoes, o hiperplano tem n-1 dimensoes. Por exemplo, com 2 atributos "
    "o hiperplano e uma reta; com 3, e um plano. A equacao e: w*x + b = 0. Pontos com w*x + b > 0 "
    "pertencem a uma classe e com w*x + b < 0 pertencem a outra."
)

pdf.subsection("Qual e o papel da margem em um SVM?")
pdf.body(
    "A margem e a distancia entre o hiperplano de separacao e os vetores de suporte de cada "
    "classe. O SVM busca maximizar essa margem, o que aumenta a capacidade de generalizacao "
    "do modelo para dados nao vistos. Uma margem maior reduz o risco de overfitting e torna "
    "o classificador mais robusto a ruido nos dados."
)

pdf.subsection("O que acontece quando os dados nao sao linearmente separaveis?")
pdf.body(
    "Quando os dados nao podem ser separados por um hiperplano linear, o SVM usa duas "
    "estrategias:\n"
    "  1) Soft Margin (parametro C): permite que alguns pontos violem a margem, tolerando "
    "erros de classificacao controlados pelo parametro C.\n"
    "  2) Kernel Trick: transforma os dados para um espaco de maior dimensionalidade onde "
    "eles se tornam linearmente separaveis, usando funcoes como RBF (Radial Basis Function), "
    "polinomial ou sigmoide, sem calcular explicitamente as coordenadas nesse espaco."
)

pdf.subsection("Qual a funcao do parametro C em um SVM?")
pdf.body(
    "C e o parametro de regularizacao que controla o trade-off entre maximizar a margem e "
    "minimizar os erros de classificacao no conjunto de treino:\n"
    "  - C grande: penaliza muito os erros, resultando em margem menor e modelo mais ajustado "
    "aos dados (risco de overfitting).\n"
    "  - C pequeno: aceita mais erros no treino em troca de uma margem maior, resultando em "
    "modelo mais generalista (risco de underfitting).\n"
    "O valor ideal de C e geralmente encontrado por validacao cruzada (cross-validation)."
)

# ── QUESTAO 2 ──────────────────────────────────────────────────────────────────
pdf.section("Questao 2 (0,5 pt) - Rede Perceptron")

pdf.subsection("Dataset e codificacao")
pdf.body(
    "Codificacao: Sim/Grandes = 1 | Nao/Pequenas = 0 | Doente = 1 | Saudavel = 0\n"
    "Parametros: taxa de aprendizado eta = 0,5 | limiar theta = -0,5 | pesos iniciais w = [0, 0, 0, 0]\n"
    "Regra de ativacao: saida = 1 se net >= theta, saida = 0 se net < theta\n"
    "Regra de atualizacao: w_j = w_j + eta * erro * x_j  (erro = alvo - saida)"
)

# Tabela do dataset
widths = [25, 20, 18, 22, 18, 25]
pdf.row(["Nome", "Febre", "Enjoo", "Manchas", "Dores", "Diagnostico"], widths, bold=True)
rows = [
    ("Joao",  "1", "1", "0 (Peq)", "1", "1 (Doente)"),
    ("Pedro", "0", "0", "1 (Gra)", "0", "0 (Saudavel)"),
    ("Maria", "1", "1", "0 (Peq)", "0", "0 (Saudavel)"),
    ("Jose",  "1", "0", "1 (Gra)", "1", "1 (Doente)"),
    ("Ana",   "1", "0", "0 (Peq)", "1", "0 (Saudavel)"),
    ("Leila", "0", "0", "1 (Gra)", "1", "1 (Doente)"),
]
for r in rows:
    pdf.row(list(r), widths)

pdf.ln(4)

pdf.subsection("Treinamento - Epoca 1 (passo a passo)")
pdf.body("Pesos iniciais: w1=0,0  w2=0,0  w3=0,0  w4=0,0")

steps_e1 = [
    ("Joao",  "[1,1,0,1]", "0,0+0,0+0,0+0,0 = 0,00", "0,00 >= -0,5", "1", "1", "0",  "sem atualizacao"),
    ("Pedro", "[0,0,1,0]", "0,0+0,0+0,0+0,0 = 0,00", "0,00 >= -0,5", "1", "0", "-1", "w3: 0+0,5*(-1)*1 = -0,5 | w=[0,0,-0.5,0]"),
    ("Maria", "[1,1,0,0]", "0*1+0*1+(-0,5)*0+0*0 = 0,00", "0,00 >= -0,5", "1", "0", "-1", "w1:-0,5 w2:-0,5 | w=[-0.5,-0.5,-0.5,0]"),
    ("Jose",  "[1,0,1,1]", "-0,5-0,5+0 = -1,00", "-1,00 < -0,5", "0", "1", "+1", "w1:0 w3:0 w4:0,5 | w=[0,-0.5,0,0.5]"),
    ("Ana",   "[1,0,0,1]", "0+0+0+0,5 = 0,50", "0,50 >= -0,5", "1", "0", "-1", "w1:-0,5 w4:0 | w=[-0.5,-0.5,0,0]"),
    ("Leila", "[0,0,1,1]", "0+0+0+0 = 0,00", "0,00 >= -0,5", "1", "1", "0",  "sem atualizacao"),
]

col_w = [16, 22, 52, 32, 14, 14, 13, 0]
col_w[-1] = 170 - sum(col_w[:-1])
pdf.set_font("Helvetica", "B", 8)
pdf.cell(16, 6, "Paciente", border=1)
pdf.cell(22, 6, "Entrada x", border=1)
pdf.cell(52, 6, "Net = sum(wi*xi)", border=1)
pdf.cell(32, 6, "Condicao", border=1)
pdf.cell(14, 6, "Saida", border=1)
pdf.cell(14, 6, "Alvo", border=1)
pdf.cell(13, 6, "Erro", border=1)
pdf.cell(col_w[-1], 6, "Atualizacao", border=1)
pdf.ln()
pdf.set_font("Helvetica", "", 7)
for s in steps_e1:
    for val, w in zip(s, col_w):
        pdf.cell(w, 6, val, border=1)
    pdf.ln()

pdf.ln(3)
pdf.set_font("Helvetica", "", 11)
pdf.body("Pesos ao final da Epoca 1: w1=-0,5 | w2=-0,5 | w3=0,0 | w4=0,0  (4 erros)")

pdf.subsection("Resumo das epocas seguintes")

col_w2 = [20, 30, 30, 30, 30, 30]
pdf.set_font("Helvetica", "B", 10)
for val, w in zip(["Epoca", "w1(Febre)", "w2(Enjoo)", "w3(Manchas)", "w4(Dores)", "Erros"], col_w2):
    pdf.cell(w, 7, val, border=1)
pdf.ln()
pdf.set_font("Helvetica", "", 10)

epocas = [
    ("1", "-0,5", "-0,5",  "0,0",  "0,0",  "4"),
    ("2", "-1,0", "-0,5", "-0,5",  "0,0",  "4"),
    ("3", "-1,0", "-0,5", "-0,5",  "0,5",  "5"),
    ("4", "-1,0", "-0,5", "-0,5",  "1,0",  "5"),
    ("5", "-1,0", "-0,5", "-0,5",  "1,0",  "3"),
    ("6+","(-1,0","(-0,5","(-0,5", "1,0)", "3 (ciclo)"),
]
for r in epocas:
    for val, w in zip(r, col_w2):
        pdf.cell(w, 7, val, border=1)
    pdf.ln()

pdf.ln(3)
pdf.set_font("Helvetica", "", 11)
pdf.body(
    "Observacao: O dataset nao e linearmente separavel com os parametros fornecidos. "
    "Isso ocorre porque Pedro [0,0,1,0] e Saudavel enquanto Jose [1,0,1,1] e Doente e Leila [0,0,1,1] "
    "e Doente - a combinacao de atributos nao permite separacao linear perfeita. "
    "O algoritmo entra em ciclo a partir da epoca 5 com pesos estabilizados em "
    "w = [-1,0; -0,5; -0,5; 1,0], que e o melhor classificador linear encontrado."
)

pdf.subsection("Pesos finais (utilizados para classificacao)")
pdf.body("w1(Febre) = -1,0 | w2(Enjoo) = -0,5 | w3(Manchas) = -0,5 | w4(Dores) = 1,0 | limiar = -0,5")

pdf.subsection("Teste de novos casos")
pdf.body(
    "Luis  (nao, nao, pequenas, sim) => x = [0, 0, 0, 1]\n"
    "  net = (-1,0)*0 + (-0,5)*0 + (-0,5)*0 + 1,0*1 = +1,0\n"
    "  1,0 >= -0,5  =>  saida = 1  =>  DOENTE\n\n"
    "Laura (sim, sim, grandes, sim) => x = [1, 1, 1, 1]\n"
    "  net = (-1,0)*1 + (-0,5)*1 + (-0,5)*1 + 1,0*1 = -1,0 + (-0,5) + (-0,5) + 1,0 = -1,0\n"
    "  -1,0 < -0,5  =>  saida = 0  =>  SAUDAVEL\n\n"
    "Analise: A classificacao de Laura como Saudavel e uma consequencia direta da nao-separabilidade "
    "linear do dataset. O peso negativo de Febre e Enjoo penaliza fortemente sua entrada, mesmo com "
    "todos os atributos iguais a 1. Para melhorar a classificacao, seria necessario usar um modelo "
    "nao-linear como MLP ou SVM com kernel RBF."
)

pdf.output(OUT)
print(f"PDF gerado: {OUT}")

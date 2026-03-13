import tf from "@tensorflow/tfjs-node";

async function trainModel(inputXs, outputYs) {
  const model = tf.sequential();

  // Primeira cama da rede:
  // entrada de 7 posições (idade normalizada + 3 cores + 3 localizacoes)

  // 80 neuronios = aqui coloquei tudo isso, pq tem pouca base de treino
  // quanto mais neuornios, mais complexidade a rede pode aprender e consequentemente
  // mais processamento ela vai usar

  // A ReLU age como um filtro: como se ela deixasse somente os dados interessantes seguirem viagem
  // na rede. Se a info chegou nesse neuronio é positiva, segue, senão descarta
  model.add(
    tf.layers.dense({ inputShape: [7], units: 80, activation: "relu" }),
  );

  // Saída: 3 neuronios - um para cada categoria | Tipo softnax que normaliza saida em probabilidade
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  // Adaptive Moment Estimation
  // é um treinador pessoal moderno para redes neurais, aprendendo com histórico de erros e acertos

  // loss categorical... compara o que o modelo acha com a resposta certa (premium será sempre [1,0,0])
  // qto mais distante da previsão, maior o erro (loss)
  // Muito usado em classificação de imgs, recomendação categorização de users.
  // Qlqr coisa que a resposta certa é apenas 1 entre várias

  model.compile({ optimizer: "adam", loss: "categoricalCrossentropy" });

  // Treinamento do modelo
  // verbose desabilita o log interno e só usa o callback
  // epochs qtde de vezes q vai rodar no dataset
  // shuffle: embaralaha pra evitar vies
  await model.fit(inputXs, outputYs, {
    verbose: 0,
    epochs: 100,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, log) =>
        console.log(`Epoch: ${epoch}: loss = ${log.loss}`),
    },
  });

  return model;
}

async function predict(model, pessoa) {
  const tfInput = tf.tensor2d(pessoa);

  const pred = model.predict(tfInput);
  const predArray = await pred.array();

  return predArray[0].map((prob, index) => ({ prob, index }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1], // Carlos
];

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
  [1, 0, 0], // premium - Erick
  [0, 1, 0], // medium - Ana
  [0, 0, 1], // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

// qto mais dados melhor
const model = await trainModel(inputXs, outputYs);

const pessoa = { nome: "zé", idade: 28, cor: "verde", localizacao: "Curitiba" };
const pessoaTensorNormalizado = [
  [
    0.2, // idade normalizada
    0, // cor azul
    0, // cor vermelho
    1, // cor verde
    0, // loc sp
    0, // log rio
    1, // loc ctb
  ],
];

const predictions = await predict(model, pessoaTensorNormalizado);
const results = predictions
  .sort((a, b) => b.prob - a.prob)
  .map((p) => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)})%`)
  .join("\n");

console.log(results);

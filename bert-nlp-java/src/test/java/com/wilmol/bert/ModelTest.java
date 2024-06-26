package com.wilmol.bert;

import static com.google.common.truth.Truth.assertThat;

import org.junit.jupiter.api.Test;

/**
 * {@link Model} tests.
 *
 * <p>Uses the example model built in the `bert-model-trainer-gradle` module.
 *
 * @author wilmol
 */
class ModelTest {

  @Test
  void testModelLoad() {
    try (Model model =
        new Model(
            "/home/will/Documents/Working/bert-nlp-java/bert-model-trainer-gradle/bert-output/1573029803",
            64,
            "/home/will/Documents/Working/bert-nlp-java/bert-model-trainer-gradle/input-data/labels.txt",
            "/home/will/Documents/Working/bert-nlp-java/bert-model-trainer-gradle/pre-trained-model/vocab.txt")) {
      assertThat(model).isNotNull();
      assertThat(model.labels()).isNotEmpty();
      assertThat(model.vocab()).isNotEmpty();
      assertThat(model.maxSeqLength()).isEqualTo(64);
    }
  }
}

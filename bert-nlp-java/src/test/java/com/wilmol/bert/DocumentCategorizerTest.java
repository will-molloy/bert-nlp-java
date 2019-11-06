package com.wilmol.bert;

import static com.google.common.truth.Truth.assertThat;

import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Document categorizer tests.
 *
 * <p>Uses the Taxi bot model built in the model-trainer module.
 *
 * @author wilmol
 */
class DocumentCategorizerTest {

  private static Model model;

  private static DocumentCategorizer documentCategorizer;

  @BeforeAll
  static void setUp() {
    model =
        new Model(
            "/home/will/Documents/Working/bert-nlp-java/bert-model-trainer-gradle/bert-output/1573029803",
            64,
            "/home/will/Documents/Working/bert-nlp-java/bert-model-trainer-gradle/input-data/labels.txt",
            "/home/will/Documents/Working/bert-nlp-java/bert-model-trainer-gradle/pre-trained-model/vocab.txt");
    Tokenizer tokenizer = new WordPieceTokenizer(model);
    FeatureExtractor featureExtractor = new FeatureExtractor(model, tokenizer);
    documentCategorizer = new DocumentCategorizer(model, featureExtractor);
  }

  @AfterAll
  static void tearDown() {
    model.close();
  }

  @ParameterizedTest
  @MethodSource("testCategorization")
  void testCategorization(String phrase, String expectedCategory) {
    List<DocumentCategorizer.LabelAndScore> list = documentCategorizer.categorize(phrase);
    assertThat(list).isNotEmpty();
    assertThat(list.get(0).label()).isEqualTo(expectedCategory);
  }

  private static Stream<Arguments> testCategorization() {
    return Stream.of(
        Arguments.of("Order me a taxi", "OrderTaxi"),
        Arguments.of("Cancel my cab", "CancelTaxi"),
        Arguments.of("Where is my taxi ?", "WhereTaxi"),
        Arguments.of("The address is 136 River Road", "GaveAddress"));
  }
}

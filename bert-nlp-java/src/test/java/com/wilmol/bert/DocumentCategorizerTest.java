package com.wilmol.bert;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Range;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * {@link DocumentCategorizer} tests.
 *
 * <p>Uses the example model built in the `bert-model-trainer-gradle` module.
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
            "/home/will/Documents/Working/bert-nlp-java/bert-model-trainer-gradle/bert-output/1573033044",
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

  @ParameterizedTest(name = "phrase={0}, expectedCategory={1}")
  @MethodSource("testCategorization")
  void testCategorization(String phrase, String expectedCategory) {
    List<DocumentCategorizer.LabelAndScore> scoredCategories =
        documentCategorizer.categorize(phrase);
    assertThat(scoredCategories).isNotEmpty();
    DocumentCategorizer.LabelAndScore bestCategory = scoredCategories.get(0);
    assertThat(bestCategory.label()).isEqualTo(expectedCategory);
    assertThat(bestCategory.predictionScore()).isIn(Range.closed(0.5f, 1.0f));
  }

  private static Stream<Arguments> testCategorization() {
    return Stream.of(
        Arguments.of("how hot is it", "CurrentWeatherIntent"),
        Arguments.of("what's the temperature", "CurrentWeatherIntent"),
        Arguments.of("how does the weather look for thursday", "FiveDayForecastIntent"),
        Arguments.of("will it rain this week", "FiveDayForecastIntent"),
        Arguments.of("will it rain this morning", "HourlyForecastIntent"),
        Arguments.of("how cold will it be this afternoon", "HourlyForecastIntent"));
  }
}

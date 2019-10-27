package com.wilmol.bert.demo;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Streams;
import com.wilmol.bert.Model;
import com.wilmol.bert.classification.TextClassifier;
import com.wilmol.bert.features.FeatureExtractor;
import com.wilmol.bert.tokenizer.WordPieceTokenizer;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;
import java.util.stream.Collectors;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Example main class for NLP text classification using BERT.
 *
 * @author wilmol
 */
public class TextClassificationDemo {

  private final Logger log = LogManager.getLogger();

  private void runDemo() {
    log.info("Loading model");
    try (Model model = new Model("", 64, "", "")) {
      log.info("Loaded model");
      WordPieceTokenizer tokenizer = new WordPieceTokenizer(model);
      FeatureExtractor featureExtractor = new FeatureExtractor(model, tokenizer);
      TextClassifier textClassifier = new TextClassifier(model, featureExtractor);

      Scanner scanner = new Scanner(System.in, StandardCharsets.UTF_8.name());
      while (true) {
        try {
          log.info("Enter text to classify, enter 'q' to quit.");
          String text = scanner.next();
          if (text.equals("q")) {
            break;
          }
          ImmutableList<TextClassifier.LabelScore> labels = textClassifier.classify(text);
          String result =
              Streams.mapWithIndex(labels.stream(), (l, i) -> (i + 1) + ": " + l)
                  .collect(Collectors.joining("\n"));
          log.info(result);
        } catch (Exception e) {
          log.error("", e);
        }
      }
      log.info("Shutting down.");
    }
  }

  public static void main(String[] args) {
    new TextClassificationDemo().runDemo();
  }
}

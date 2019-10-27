package com.wilmol.bert.tokenizer;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.CharMatcher;
import com.google.common.collect.ImmutableList;
import com.wilmol.bert.Model;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Word piece tokenizer. Based on: <a
 * href=https://github.com/google-research/bert/blob/master/tokenization.py>https://github.com/google-research/bert/blob/master/tokenization.py</a>
 *
 * @author wilmol
 */
public class WordPieceTokenizer implements Tokenizer {

  private static final String CLS_TOKEN = "[CLS]";

  private static final String UNK_TOKEN = "[UNK]";

  private static final String SEP_TOKEN = "[SEP]";

  private static final int MAX_INPUT_CHARS_PER_WORD = 30;

  private final Model model;

  public WordPieceTokenizer(Model model) {
    this.model = checkNotNull(model);
  }

  /**
   * Tokenizes a piece of text into its word pieces.
   *
   * <p>This uses a greedy longest-match-first algorithm to perform tokenization using the given
   * vocabulary.
   *
   * <p>Note if the given text produces more tokens than the models max sequence length, it'll be
   * truncated on the right.
   *
   * @param text A single token or whitespace separated tokens.
   * @return A list of wordpiece tokens.
   */
  @Override
  public ImmutableList<String> tokenize(String text) {
    String splitText = splitOnPunctuation(text.trim().toLowerCase(Locale.US));

    List<String> outputTokens = new ArrayList<>();

    outputTokens.add(CLS_TOKEN);

    String[] tokens = splitText.split(" ");

    outer:
    for (String token : tokens) {
      if (token.length() > MAX_INPUT_CHARS_PER_WORD) {
        outputTokens.add(UNK_TOKEN);
        if (outputTokens.size() == model.maxSeqLength() - 1) {
          break outer;
        }
      } else {
        StringBuilder chars = new StringBuilder(token);
        int start = 0;

        while (start < chars.length()) {
          int end = chars.length();
          String currentSubString = null;

          while (start < end) {
            String subString = chars.substring(start, end);

            if (start > 0) {
              subString = "##" + subString;
            }

            if (model.vocab().containsKey(subString)) {
              currentSubString = subString;
              break;
            }
            end -= 1;
          }

          if (currentSubString != null) {
            outputTokens.add(currentSubString);
            if (outputTokens.size() == model.maxSeqLength() - 1) {
              break outer;
            }
            start = end;
          } else {
            outputTokens.add(UNK_TOKEN);
            if (outputTokens.size() == model.maxSeqLength() - 1) {
              break outer;
            }
            break;
          }
        }
      }
    }

    outputTokens.add(SEP_TOKEN);

    return ImmutableList.copyOf(outputTokens);
  }

  private static String splitOnPunctuation(String text) {
    char[] chars = text.toCharArray();

    int i = 0;
    StringBuilder output = new StringBuilder();

    while (i < chars.length) {
      char currChar = chars[i];

      if (isPunctuation(currChar)) {
        output.append(" ").append(currChar).append(" ");
      } else {
        output.append(currChar);
      }

      i += 1;
    }
    return output.toString();
  }

  private static boolean isPunctuation(char cp) {
    // treat all non ascii as punctuation
    if (!CharMatcher.ascii().matches(cp)) {
      return true;
    }
    return (cp >= 33 && cp <= 47)
        || (cp >= 58 && cp <= 64)
        || (cp >= 91 && cp <= 96)
        || (cp >= 123 && cp <= 126);
  }
}

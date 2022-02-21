import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_three_datasets(df):
    # Original Dataset
    df_og = df[df['OG'] == 'Y']
    train_og, test_og = train_test_split(df_og, test_size=0.3, random_state=11)

    df_og_01_text_train = train_og.filter(['id', 'OG', 'text', 'target_word_og', 'sentiment'], axis=1)
    df_og_01_text_train = df_og_01_text_train.rename(columns={"text": "text", "target_word_og": "target"})
    df_og_01_text_train.to_csv('..//final_datasets/og/df_og_01_text_train.csv')
    df_og_01_text_test = test_og.filter(['id', 'OG', 'text', 'target_word_og', 'sentiment'], axis=1)
    df_og_01_text_test = df_og_01_text_test.rename(columns={"text": "text", "target_word_og": "target"})
    df_og_01_text_test.to_csv('..//final_datasets/og/df_og_01_text_test.csv')

    df_og_02_textoneline_train = train_og.filter(['id', 'OG', 'text_oneline', 'target_word_og', 'sentiment'], axis=1)
    df_og_02_textoneline_train = df_og_02_textoneline_train.rename(
        columns={"text_oneline": "text", "target_word_og": "target"})
    df_og_02_textoneline_train.to_csv('..//final_datasets/og/df_og_02_textoneline_train.csv')
    df_og_02_textoneline_test = test_og.filter(['id', 'OG', 'text_oneline', 'target_word_og', 'sentiment'], axis=1)
    df_og_02_textoneline_test = df_og_02_textoneline_test.rename(
        columns={"text_oneline": "text", "target_word_og": "target"})
    df_og_02_textoneline_test.to_csv('..//final_datasets/og/df_og_02_textoneline_test.csv')

    df_og_03_textmasked_train = train_og.filter(['id', 'OG', 'text_masked', 'target_word_og', 'sentiment'], axis=1)
    df_og_03_textmasked_train = df_og_03_textmasked_train.rename(
        columns={"text_masked": "text", "target_word_og": "target"})
    df_og_03_textmasked_train.to_csv('..//final_datasets/og/df_og_03_textmasked_train.csv')
    df_og_03_textmasked_test = test_og.filter(['id', 'OG', 'text_masked', 'target_word_og', 'sentiment'], axis=1)
    df_og_03_textmasked_test = df_og_03_textmasked_test.rename(
        columns={"text_masked": "text", "target_word_og": "target"})
    df_og_03_textmasked_test.to_csv('..//final_datasets/og/df_og_03_textmasked_test.csv')

    df_og_04_textwithspaces_train = train_og.filter(['id', 'OG', 'text_withspaces', 'target_word_og', 'sentiment'],
                                                    axis=1)
    df_og_04_textwithspaces_train = df_og_04_textwithspaces_train.rename(
        columns={"text_withspaces": "text", "target_word_og": "target"})
    df_og_04_textwithspaces_train.to_csv('..//final_datasets/og/df_og_04_textwithspaces_train.csv')
    df_og_04_textwithspaces_test = test_og.filter(['id', 'OG', 'text_withspaces', 'target_word_og', 'sentiment'],
                                                  axis=1)
    df_og_04_textwithspaces_test = df_og_04_textwithspaces_test.rename(
        columns={"text_withspaces": "text", "target_word_og": "target"})
    df_og_04_textwithspaces_test.to_csv('..//final_datasets/og/df_og_04_textwithspaces_test.csv')

    df_og_05_textlowered_train = train_og.filter(['id', 'OG', 'text_lowered', 'target_lowered', 'sentiment'], axis=1)
    df_og_05_textlowered_train = df_og_05_textlowered_train.rename(
        columns={"text_lowered": "text", "target_lowered": "target"})
    df_og_05_textlowered_train.to_csv('..//final_datasets/og/df_og_05_textlowered_train.csv')
    df_og_05_textlowered_test = test_og.filter(['id', 'OG', 'text_lowered', 'target_lowered', 'sentiment'], axis=1)
    df_og_05_textlowered_test = df_og_05_textlowered_test.rename(
        columns={"text_lowered": "text", "target_lowered": "target"})
    df_og_05_textlowered_test.to_csv('..//final_datasets/og/df_og_05_textlowered_test.csv')

    df_og_06_textremovedspecialchars_train = train_og.filter(
        ['id', 'OG', 'text_removedspecialchars', 'target_lowered', 'sentiment'], axis=1)
    df_og_06_textremovedspecialchars_train = df_og_06_textremovedspecialchars_train.rename(
        columns={"text_removedspecialchars": "text", "target_lowered": "target"})
    df_og_06_textremovedspecialchars_train.to_csv(
        '..//final_datasets/og/df_og_06_textremovedspecialchars_train.csv')
    df_og_06_textremovedspecialchars_test = test_og.filter(
        ['id', 'OG', 'text_removedspecialchars', 'target_lowered', 'sentiment'], axis=1)
    df_og_06_textremovedspecialchars_test = df_og_06_textremovedspecialchars_test.rename(
        columns={"text_removedspecialchars": "text", "target_lowered": "target"})
    df_og_06_textremovedspecialchars_test.to_csv(
        '..//final_datasets/og/df_og_06_textremovedspecialchars_test.csv')

    df_og_07_textlowerednoaccent_train = train_og.filter(['id', 'OG', 'text_noaccent', 'target_noaccent', 'sentiment'],
                                                         axis=1)
    df_og_07_textlowerednoaccent_train = df_og_07_textlowerednoaccent_train.rename(
        columns={"text_noaccent": "text", "target_noaccent": "target"})
    df_og_07_textlowerednoaccent_train.to_csv(
        '..//final_datasets/og/df_og_07_textlowerednoaccent_train.csv')
    df_og_07_textlowerednoaccent_test = test_og.filter(['id', 'OG', 'text_noaccent', 'target_noaccent', 'sentiment'],
                                                       axis=1)
    df_og_07_textlowerednoaccent_test = df_og_07_textlowerednoaccent_test.rename(
        columns={"text_noaccent": "text", "target_noaccent": "target"})
    df_og_07_textlowerednoaccent_test.to_csv(
        '..//final_datasets/og/df_og_07_textlowerednoaccent_test.csv')

    df_og_08_textlowerednoaccentremovedspecialchars_train = train_og.filter(
        ['id', 'OG', 'text_noaccentnospecials', 'target_noaccent', 'sentiment'], axis=1)
    df_og_08_textlowerednoaccentremovedspecialchars_train = df_og_08_textlowerednoaccentremovedspecialchars_train.rename(
        columns={"text_noaccentnospecials": "text", "target_noaccent": "target"})
    df_og_08_textlowerednoaccentremovedspecialchars_train.to_csv(
        '..//final_datasets/og/df_og_08_textlowerednoaccentremovedspecialchars_train.csv')
    df_og_08_textlowerednoaccentremovedspecialchars_test = test_og.filter(
        ['id', 'OG', 'text_noaccentnospecials', 'target_noaccent', 'sentiment'], axis=1)
    df_og_08_textlowerednoaccentremovedspecialchars_test = df_og_08_textlowerednoaccentremovedspecialchars_test.rename(
        columns={"text_noaccentnospecials": "text", "target_noaccent": "target"})
    df_og_08_textlowerednoaccentremovedspecialchars_test.to_csv(
        '..//final_datasets/og/df_og_08_textlowerednoaccentremovedspecialchars_test.csv')

    # Oversampled Dataset
    train_xx, test_xx = train_test_split(df, test_size=0.3, random_state=11)

    df_xx_01_text_train = train_xx.filter(['id', 'OG', 'text', 'target_word_og', 'sentiment'], axis=1)
    df_xx_01_text_train = df_xx_01_text_train.rename(columns={"text": "text", "target_word_og": "target"})
    df_xx_01_text_train.to_csv('..//final_datasets/xx/df_xx_01_text_train.csv')
    df_xx_01_text_test = test_xx.filter(['id', 'OG', 'text', 'target_word_og', 'sentiment'], axis=1)
    df_xx_01_text_test = df_xx_01_text_test.rename(columns={"text": "text", "target_word_og": "target"})
    df_xx_01_text_test.to_csv('..//final_datasets/xx/df_xx_01_text_test.csv')

    df_xx_02_textoneline_train = train_xx.filter(['id', 'OG', 'text_oneline', 'target_word_og', 'sentiment'], axis=1)
    df_xx_02_textoneline_train = df_xx_02_textoneline_train.rename(
        columns={"text_oneline": "text", "target_word_og": "target"})
    df_xx_02_textoneline_train.to_csv('..//final_datasets/xx/df_xx_02_textoneline_train.csv')
    df_xx_02_textoneline_test = test_xx.filter(['id', 'OG', 'text_oneline', 'target_word_og', 'sentiment'], axis=1)
    df_xx_02_textoneline_test = df_xx_02_textoneline_test.rename(
        columns={"text_oneline": "text", "target_word_og": "target"})
    df_xx_02_textoneline_test.to_csv('..//final_datasets/xx/df_xx_02_textoneline_test.csv')

    df_xx_03_textmasked_train = train_xx.filter(['id', 'OG', 'text_masked', 'target_word_og', 'sentiment'], axis=1)
    df_xx_03_textmasked_train = df_xx_03_textmasked_train.rename(
        columns={"text_masked": "text", "target_word_og": "target"})
    df_xx_03_textmasked_train.to_csv('..//final_datasets/xx/df_xx_03_textmasked_train.csv')
    df_xx_03_textmasked_test = test_xx.filter(['id', 'OG', 'text_masked', 'target_word_og', 'sentiment'], axis=1)
    df_xx_03_textmasked_test = df_xx_03_textmasked_test.rename(
        columns={"text_masked": "text", "target_word_og": "target"})
    df_xx_03_textmasked_test.to_csv('..//final_datasets/xx/df_xx_03_textmasked_test.csv')

    df_xx_04_textwithspaces_train = train_xx.filter(['id', 'OG', 'text_withspaces', 'target_word_og', 'sentiment'],
                                                    axis=1)
    df_xx_04_textwithspaces_train = df_xx_04_textwithspaces_train.rename(
        columns={"text_withspaces": "text", "target_word_og": "target"})
    df_xx_04_textwithspaces_train.to_csv('..//final_datasets/xx/df_xx_04_textwithspaces_train.csv')
    df_xx_04_textwithspaces_test = test_xx.filter(['id', 'OG', 'text_withspaces', 'target_word_og', 'sentiment'],
                                                  axis=1)
    df_xx_04_textwithspaces_test = df_xx_04_textwithspaces_test.rename(
        columns={"text_withspaces": "text", "target_word_og": "target"})
    df_xx_04_textwithspaces_test.to_csv('..//final_datasets/xx/df_xx_04_textwithspaces_test.csv')

    df_xx_05_textlowered_train = train_xx.filter(['id', 'OG', 'text_lowered', 'target_lowered', 'sentiment'], axis=1)
    df_xx_05_textlowered_train = df_xx_05_textlowered_train.rename(
        columns={"text_lowered": "text", "target_lowered": "target"})
    df_xx_05_textlowered_train.to_csv('..//final_datasets/xx/df_xx_05_textlowered_train.csv')
    df_xx_05_textlowered_test = test_xx.filter(['id', 'OG', 'text_lowered', 'target_lowered', 'sentiment'], axis=1)
    df_xx_05_textlowered_test = df_xx_05_textlowered_test.rename(
        columns={"text_lowered": "text", "target_lowered": "target"})
    df_xx_05_textlowered_test.to_csv('..//final_datasets/xx/df_xx_05_textlowered_test.csv')

    df_xx_06_textremovedspecialchars_train = train_xx.filter(
        ['id', 'OG', 'text_removedspecialchars', 'target_lowered', 'sentiment'], axis=1)
    df_xx_06_textremovedspecialchars_train = df_xx_06_textremovedspecialchars_train.rename(
        columns={"text_removedspecialchars": "text", "target_lowered": "target"})
    df_xx_06_textremovedspecialchars_train.to_csv(
        '..//final_datasets/xx/df_xx_06_textremovedspecialchars_train.csv')
    df_xx_06_textremovedspecialchars_test = test_xx.filter(
        ['id', 'OG', 'text_removedspecialchars', 'target_lowered', 'sentiment'], axis=1)
    df_xx_06_textremovedspecialchars_test = df_xx_06_textremovedspecialchars_test.rename(
        columns={"text_removedspecialchars": "text", "target_lowered": "target"})
    df_xx_06_textremovedspecialchars_test.to_csv(
        '..//final_datasets/xx/df_xx_06_textremovedspecialchars_test.csv')

    df_xx_07_textlowerednoaccent_train = train_xx.filter(['id', 'OG', 'text_noaccent', 'target_noaccent', 'sentiment'],
                                                         axis=1)
    df_xx_07_textlowerednoaccent_train = df_xx_07_textlowerednoaccent_train.rename(
        columns={"text_noaccent": "text", "target_noaccent": "target"})
    df_xx_07_textlowerednoaccent_train.to_csv(
        '..//final_datasets/xx/df_xx_07_textlowerednoaccent_train.csv')
    df_xx_07_textlowerednoaccent_test = test_xx.filter(['id', 'OG', 'text_noaccent', 'target_noaccent', 'sentiment'],
                                                       axis=1)
    df_xx_07_textlowerednoaccent_test = df_xx_07_textlowerednoaccent_test.rename(
        columns={"text_noaccent": "text", "target_noaccent": "target"})
    df_xx_07_textlowerednoaccent_test.to_csv(
        '..//final_datasets/xx/df_xx_07_textlowerednoaccent_test.csv')

    df_xx_08_textlowerednoaccentremovedspecialchars_train = train_xx.filter(
        ['id', 'OG', 'text_noaccentnospecials', 'target_noaccent', 'sentiment'], axis=1)
    df_xx_08_textlowerednoaccentremovedspecialchars_train = df_xx_08_textlowerednoaccentremovedspecialchars_train.rename(
        columns={"text_noaccentnospecials": "text", "target_noaccent": "target"})
    df_xx_08_textlowerednoaccentremovedspecialchars_train.to_csv(
        '..//final_datasets/xx/df_xx_08_textlowerednoaccentremovedspecialchars_train.csv')
    df_xx_08_textlowerednoaccentremovedspecialchars_test = test_xx.filter(
        ['id', 'OG', 'text_noaccentnospecials', 'target_noaccent', 'sentiment'], axis=1)
    df_xx_08_textlowerednoaccentremovedspecialchars_test = df_xx_08_textlowerednoaccentremovedspecialchars_test.rename(
        columns={"text_noaccentnospecials": "text", "target_noaccent": "target"})
    df_xx_08_textlowerednoaccentremovedspecialchars_test.to_csv(
        '..//final_datasets/xx/df_xx_08_textlowerednoaccentremovedspecialchars_test.csv')

    # Hybrid Dataset
    df_0 = df[df['sentiment'] == 2]
    df_0 = df_0.sample(frac=1).reset_index(drop=True)
    df_0 = df_0.head(1862)

    df_rest = df[df['sentiment'] != 2]
    df_15 = df_rest.append(df_0)
    df_15 = df_15.sample(frac=1).reset_index(drop=True)

    train_15, test_15 = train_test_split(df_15, test_size=0.3, random_state=11)

    df_15_01_text_train = train_15.filter(['id', 'OG', 'text', 'target_word_og', 'sentiment'], axis=1)
    df_15_01_text_train = df_15_01_text_train.rename(columns={"text": "text", "target_word_og": "target"})
    df_15_01_text_train.to_csv('..//final_datasets/15/df_15_01_text_train.csv')
    df_15_01_text_test = test_15.filter(['id', 'OG', 'text', 'target_word_og', 'sentiment'], axis=1)
    df_15_01_text_test = df_15_01_text_test.rename(columns={"text": "text", "target_word_og": "target"})
    df_15_01_text_test.to_csv('..//final_datasets/15/df_15_01_text_test.csv')

    df_15_02_textoneline_train = train_15.filter(['id', 'OG', 'text_oneline', 'target_word_og', 'sentiment'], axis=1)
    df_15_02_textoneline_train = df_15_02_textoneline_train.rename(
        columns={"text_oneline": "text", "target_word_og": "target"})
    df_15_02_textoneline_train.to_csv('..//final_datasets/15/df_15_02_textoneline_train.csv')
    df_15_02_textoneline_test = test_15.filter(['id', 'OG', 'text_oneline', 'target_word_og', 'sentiment'], axis=1)
    df_15_02_textoneline_test = df_15_02_textoneline_test.rename(
        columns={"text_oneline": "text", "target_word_og": "target"})
    df_15_02_textoneline_test.to_csv('..//final_datasets/15/df_15_02_textoneline_test.csv')

    df_15_03_textmasked_train = train_15.filter(['id', 'OG', 'text_masked', 'target_word_og', 'sentiment'], axis=1)
    df_15_03_textmasked_train = df_15_03_textmasked_train.rename(
        columns={"text_masked": "text", "target_word_og": "target"})
    df_15_03_textmasked_train.to_csv('..//final_datasets/15/df_15_03_textmasked_train.csv')
    df_15_03_textmasked_test = test_15.filter(['id', 'OG', 'text_masked', 'target_word_og', 'sentiment'], axis=1)
    df_15_03_textmasked_test = df_15_03_textmasked_test.rename(
        columns={"text_masked": "text", "target_word_og": "target"})
    df_15_03_textmasked_test.to_csv('..//final_datasets/15/df_15_03_textmasked_test.csv')

    df_15_04_textwithspaces_train = train_15.filter(['id', 'OG', 'text_withspaces', 'target_word_og', 'sentiment'],
                                                    axis=1)
    df_15_04_textwithspaces_train = df_15_04_textwithspaces_train.rename(
        columns={"text_withspaces": "text", "target_word_og": "target"})
    df_15_04_textwithspaces_train.to_csv('..//final_datasets/15/df_15_04_textwithspaces_train.csv')
    df_15_04_textwithspaces_test = test_15.filter(['id', 'OG', 'text_withspaces', 'target_word_og', 'sentiment'],
                                                  axis=1)
    df_15_04_textwithspaces_test = df_15_04_textwithspaces_test.rename(
        columns={"text_withspaces": "text", "target_word_og": "target"})
    df_15_04_textwithspaces_test.to_csv('..//final_datasets/15/df_15_04_textwithspaces_test.csv')

    df_15_05_textlowered_train = train_15.filter(['id', 'OG', 'text_lowered', 'target_lowered', 'sentiment'], axis=1)
    df_15_05_textlowered_train = df_15_05_textlowered_train.rename(
        columns={"text_lowered": "text", "target_lowered": "target"})
    df_15_05_textlowered_train.to_csv('..//final_datasets/15/df_15_05_textlowered_train.csv')
    df_15_05_textlowered_test = test_15.filter(['id', 'OG', 'text_lowered', 'target_lowered', 'sentiment'], axis=1)
    df_15_05_textlowered_test = df_15_05_textlowered_test.rename(
        columns={"text_lowered": "text", "target_lowered": "target"})
    df_15_05_textlowered_test.to_csv('..//final_datasets/15/df_15_05_textlowered_test.csv')

    df_15_06_textremovedspecialchars_train = train_15.filter(
        ['id', 'OG', 'text_removedspecialchars', 'target_lowered', 'sentiment'], axis=1)
    df_15_06_textremovedspecialchars_train = df_15_06_textremovedspecialchars_train.rename(
        columns={"text_removedspecialchars": "text", "target_lowered": "target"})
    df_15_06_textremovedspecialchars_train.to_csv(
        '..//final_datasets/15/df_15_06_textremovedspecialchars_train.csv')
    df_15_06_textremovedspecialchars_test = test_15.filter(
        ['id', 'OG', 'text_removedspecialchars', 'target_lowered', 'sentiment'], axis=1)
    df_15_06_textremovedspecialchars_test = df_15_06_textremovedspecialchars_test.rename(
        columns={"text_removedspecialchars": "text", "target_lowered": "target"})
    df_15_06_textremovedspecialchars_test.to_csv(
        '..//final_datasets/15/df_15_06_textremovedspecialchars_test.csv')

    df_15_07_textlowerednoaccent_train = train_15.filter(['id', 'OG', 'text_noaccent', 'target_noaccent', 'sentiment'],
                                                         axis=1)
    df_15_07_textlowerednoaccent_train = df_15_07_textlowerednoaccent_train.rename(
        columns={"text_noaccent": "text", "target_noaccent": "target"})
    df_15_07_textlowerednoaccent_train.to_csv(
        '..//final_datasets/15/df_15_07_textlowerednoaccent_train.csv')
    df_15_07_textlowerednoaccent_test = test_15.filter(['id', 'OG', 'text_noaccent', 'target_noaccent', 'sentiment'],
                                                       axis=1)
    df_15_07_textlowerednoaccent_test = df_15_07_textlowerednoaccent_test.rename(
        columns={"text_noaccent": "text", "target_noaccent": "target"})
    df_15_07_textlowerednoaccent_test.to_csv(
        '..//final_datasets/15/df_15_07_textlowerednoaccent_test.csv')

    df_15_08_textlowerednoaccentremovedspecialchars_train = train_15.filter(
        ['id', 'OG', 'text_noaccentnospecials', 'target_noaccent', 'sentiment'], axis=1)
    df_15_08_textlowerednoaccentremovedspecialchars_train = df_15_08_textlowerednoaccentremovedspecialchars_train.rename(
        columns={"text_noaccentnospecials": "text", "target_noaccent": "target"})
    df_15_08_textlowerednoaccentremovedspecialchars_train.to_csv(
        '..//final_datasets/15/df_15_08_textlowerednoaccentremovedspecialchars_train.csv')
    df_15_08_textlowerednoaccentremovedspecialchars_test = test_15.filter(
        ['id', 'OG', 'text_noaccentnospecials', 'target_noaccent', 'sentiment'], axis=1)
    df_15_08_textlowerednoaccentremovedspecialchars_test = df_15_08_textlowerednoaccentremovedspecialchars_test.rename(
        columns={"text_noaccentnospecials": "text", "target_noaccent": "target"})
    df_15_08_textlowerednoaccentremovedspecialchars_test.to_csv(
        '..//final_datasets/15/df_15_08_textlowerednoaccentremovedspecialchars_test.csv')
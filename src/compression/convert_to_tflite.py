from tensorflow_lite_support.metadata.metadata_schema_py_generated import AssociatedFile, AssociatedFileT, FeatureProperties, SentencePieceTokenizerOptions
from transformers import AutoTokenizer
import transformers
from src.dataset.wic_dataset import *
from transformers import AutoTokenizer
from src.models.sentence_encoder import OnnxSentenceTransformerWrapper, SentenceTransformerWrapper
from src.models.modeling import TransformerWrapper, OnnxTransformerWrapper
import argparse
from src.dataset.parallel_dataset import *
from src.configurations import config
from src.modules.model_compression import convert_to_onnx
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
import os


def convert_to_tf(onnx_path: str, tf_path: str):
    if not os.path.exists(tf_path):
        os.makedirs(tf_path)
    onnx_model = onnx.load(onnx_path)
    #onnx.checker.check_model(onnx_model)  # Checks signature
    tf_rep = prepare(onnx_model)  # Prepare TF representation
    tf_rep.export_graph(tf_path)
    return tf_path

def tf_to_tf_lite(tf_path, tf_lite_path):
    if not os.path.exists(tf_lite_path):
        os.makedirs(tf_lite_path)
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path) 
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()  # Creates converter instance
    tflite_model_path = tf_lite_path + "/tflite-model"
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    return tflite_model_path



def build_metadata(
    model_file_path: str,
    #model_name: str,
    #model_description: str,
    #model_author: str,
    #model_license: str,
    vocab_file_path: str=None, 
    sentencepiece_model_path: str=None):


    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = 'distilbert news'
    model_meta.description = 'some model description'
    model_meta.version = 'v1'
    model_meta.author = 'Unknown'
    model_meta.license = 'Apache License. Version 2.0'
    model_meta.minParserVersion = '1.1.0'

    # Creates input info.
    ids = _metadata_fb.TensorMetadataT()
    segment_ids = _metadata_fb.TensorMetadataT()
    mask = _metadata_fb.TensorMetadataT()
    
    ids.name = "ids"
    ids.description = "Tokenized ids of input text."
    ids.content = _metadata_fb.ContentT()
    ids.content.contentProperties = _metadata_fb.FeaturePropertiesT()
    ids.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)

    segment_ids.name = "segment_ids"
    segment_ids.description = "0 for the first sequence, 1 for the second sequence if exits."
    segment_ids.content = _metadata_fb.ContentT()
    segment_ids.content.contentProperties = _metadata_fb.FeaturePropertiesT()
    segment_ids.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)

    mask.name = "mask"
    mask.description = "Mask with 1 for real tokens and 0 for padding tokens."
    mask.content = _metadata_fb.ContentT()
    mask.content.contentProperties = _metadata_fb.FeaturePropertiesT()
    mask.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    

    # Creates output info.
    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.name = "output"
    output_meta.description = "the id of the output class"
    output_meta.content = _metadata_fb.ContentT()
    output_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
    output_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)

    """
    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = os.path.basename(label_file_path)
    label_file.description = "Labels for the categories to be predicted."
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
    output_meta.associatedFiles = [label_file]
    """
    
    input_process_units = []
    if sentencepiece_model_path is not None:
    #Add sentencepiece specific process unit
        sentencepiece_process_unit = _metadata_fb.ProcessUnitT()
        sentencepiece_process_unit.optionsType = (
            _metadata_fb.ProcessUnitOptions.SentencePieceTokenizerOptions)
        sentencepiece_process_unit.options = _metadata_fb.SentencePieceTokenizerOptionsT()
        sentencepiece_model = AssociatedFileT()
        sentencepiece_model.name="30k-clean-model",
        sentencepiece_model.description="The sentence piece model file."
        sentencepiece_process_unit.options.sentencePieceModel = [sentencepiece_model]
        input_process_units.append(sentencepiece_process_unit)

    if vocab_file_path is not None:
        model_process_unit = _metadata_fb.ProcessUnitT()
        model_process_unit.optionsType = (
            _metadata_fb.ProcessUnitOptions.BertTokenizerOptions)
        model_process_unit.options = _metadata_fb.BertTokenizerOptionsT()
        vocab_file = AssociatedFileT()
        vocab_file.name="jp word piece vocab",
        vocab_file.description="Japanese Vocabulary file for the BERT tokenizer.",
        vocab_file.type=_metadata_fb.AssociatedFileType.VOCABULARY
        model_process_unit.options.vocabFile = [vocab_file]
        input_process_units.append(model_process_unit)

    #Put metadata together

    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [ids, mask, segment_ids]
    subgraph.outputTensorMetadata = [output_meta]
    
    subgraph.inputProcessUnits = input_process_units

    model_meta.subgraphMetadata = [subgraph]

    #create flat buffers for metadata
    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()

    #associate files to metadata

    populator = _metadata.MetadataPopulator.with_model_file(model_file_path)
    populator.load_metadata_buffer(metadata_buf)
    if vocab_file_path is not None:
        files = [vocab_file_path]
    else:
        files = [sentencepiece_model_path]
    populator.load_associated_files(files)
    populator.populate()
    


if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, dest="config_name")
        parser.add_argument('--model', type=str, dest="model", default="bandainamco-mirai/distilbert-base-japanese")
        parser.add_argument('--model_path', type=str, dest="model_path", default="../training/trained_models/distilbert-japanese-nikkei/")
        parser.add_argument('--save_path', dest="save_path", type=str, default="./output/onnx")
        parser.add_argument('--tf_path', dest="tf_path", type=str, default="./output/tf")
        parser.add_argument('--tf_lite_path', dest="tf_lite_path", type=str, default="./output/tflite")
        parser.add_argument('--spiece_path', dest="sentencepiece_path", type=str, default="./output/tflite")

        args = parser.parse_args()
     
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        model_config = config.ModelParameters(
        model_name = args.config_name
        )

        configuration = config.Configuration(
            model_parameters=model_config,
            model = args.model,
            save_path = args.save_path,
            tokenizer=tokenizer
        )

        model_to_convert = OnnxTransformerWrapper.load_pretrained(args.model_path, params=configuration)

        symbolic_names = {
                0: 'batch',
                1: 'sequence'
            }
        
        symbolic_output_names = {
                0: 'batch'
            }
        
        if "distilbert" in args.model:
            dynamic_axes = {
                'attention_mask': symbolic_names,
                'input_ids': symbolic_names,
            }
        else:
            dynamic_axes = {
                'attention_mask': symbolic_names,
                'input_ids': symbolic_names,
                'segment_ids': symbolic_output_names,
            }

        output_names = ['category']


        print("Converting model to Onnx format...")
        onnx_path = convert_to_onnx(
            model_to_convert, 
            configuration, 
            dynamic_axes=dynamic_axes, 
            output_names=output_names, 
            quantize=True,
            has_token_type_ids = not "distilbert" in args.model)
        print("Done")
        print("Converting model to tensorflow...")
        tf_path = convert_to_tf(onnx_path, args.tf_path + "/" + args.config_name)
        print("Done")
        print("Converting model to tf-lite")
        tflite_path = args.tf_lite_path + "/" + args.config_name
        tflite_model_path = tf_to_tf_lite(tf_path, tflite_path)
        print("Done")
        print("Building model metadata")
        build_metadata_test(vocab_file_path=args.model_path + "vocab.txt", model_file_path = tflite_model_path)


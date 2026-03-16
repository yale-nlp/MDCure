import torch
import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from ast import literal_eval
import pandas as pd 
from openai import OpenAI

# set up OpenAI client
client = OpenAI()

# style-specific template
TEMPLATE = """You're proficient in crafting complex questions. Generate only one question and one answer that adheres to the provided #Articles#. Make sure the question and answer are factually consistent with the #Articles#. The question should meet the following criteria: 
0. The person answering the question cannot see the #Articles#, so the question must not contain phrases like 'Given the information provided', 'Based on the provided information', or similar expressions that imply direct citations or references from #Articles#. 
1. The question must REQUIRE synthesis of information in at least 2 of the provided documents in order to answer correctly. The more documents are involved the better. Ideally all documents are required to answer the question, such that losing any one of them will lead person answering the question to provide an incorrect response. You will lose your job if this criterion is not satisfied.
2. {characteristic}
3. {type}
4. {style}.
5. It requires {answer_length} to answer correctly.

The answer must be {answer_length} in length.

### Articles: 
{docs}

Question: <respond here>
Answer: <respond here>"""

characteristics = [
    "It should be complex and require multiple-step reasoning across the documents to solve.",
    "It demands critical thinking skills to analyze, evaluate, and synthesize multiple pieces of information from the different documents.",
    "It demands integrating knowledge from multiple documents to address its multifaceted nature.",
    "It should be simple and require only a few words to answer, yet utilize supporting evidence from at least 2 documents."
]
types = [
    "It is a Natural language inference question: Assessing if evidence supports a conclusion.",
    "It is a Paraphrasing question: Rewording a statement while retaining its meaning.",
    "It is a Summarization question: Condensing key information from a larger text.",
    "It is an Informational question: Locating a specific piece of information in the given evidence."
]
styles = [
    "It should be in the style of a command or imperative. For example, 'Write a paragraph about...' or 'Describe the...'",
    "It should be in the style of a question or interrogative. For example, 'What is the..?' or 'How do you...?'",
    "It should be in the style of a short phrase that serves as a query. For example, 'today's forecast.' or 'Donna’s car accident.'"
]
answer_lengths = [
    "1-2 words",
    "3-4 words",
    "a phrase of at least 5-6 words",
    "1-2 sentences",
    "3-4 sentences",
    "6 sentences",
    "8 sentences",
    "10 sentences",
]

def gpt_generate(model, prompt=None, messages=None):
    if messages:
        msg = messages
    else:
        msg = [
            {"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": prompt},
        ]
    return client.chat.completions.create(
                messages=msg,
                model=model,
                n=1,
                temperature=1,
            )

# general prompts G-N
def get_general_prompt(prompt_num, docs):

    docs_str = ""
    for doc in docs:
        docs_str += doc.replace("\n"," ")
        docs_str += "\n\n" 

    if prompt_num==1: # prompt template G
        prompt = f"{docs_str}A question or command that can ONLY be answered by utilizing ALL of the above documents and that CANNOT be answered if any one document is removed is:\nQuestion: <respond here>\nAnswer: <respond here briefly>" 

    elif prompt_num==2: # prompt template H
        prompt = f"{docs_str}What is a question or command that can ONLY be answered by utilizing ALL of the above documents and that CANNOT be answered if any one document is removed?\nQuestion: <respond here>\nAnswer: <respond here briefly>" 

    elif prompt_num==3: # prompt template I
        prompt = f"Articles:\n\n{docs_str}What is an exam question that can ONLY be answered by utilizing ALL of the above documents and that CANNOT be answered if any one document is removed?\n\nExam Question: <respond here>\nAnswer: <respond here briefly>" 

    elif prompt_num==4: # prompt template J
        prompt = f"{docs_str}What is a question or command that can ONLY be answered by utilizing ALL of the above documents and that CANNOT be answered if any one document is removed?\nQuestion: <respond here>\nAnswer: <respond here, feel free to use a single word or phrase>" 

    elif prompt_num==5: # prompt template K
        prompt = f"{docs_str}A question or command that can ONLY be answered by utilizing ALL of the above documents and that CANNOT be answered if any one document is removed is:\nQuestion: <respond here>\nAnswer: <respond here>"

    elif prompt_num==6: # prompt template L
        prompt = f"{docs_str}What is a question or command that can ONLY be answered by utilizing ALL of the above documents and that CANNOT be answered if any one document is removed?\nQuestion: <respond here>\nAnswer: <respond here, using ONLY a single word or phrase>" 

    elif prompt_num==7: # prompt template M
        prompt = f"Articles:\n\n{docs_str}Contrasting Question: <respond here>\nAnswer: <respond here briefly>" 

    elif prompt_num==8: # prompt template N
        prompt = f"Articles:\n\n{docs_str}What is an exam question that can ONLY be answered by utilizing ALL of the above documents and that CANNOT be answered if any one document is removed?\n\nExam Question: <respond here>\nAnswer Choices: <respond here>\nAnswer: <answer letter only>" 

    return prompt

def generate_and_save_instructions(args):

    def get_template_prompt(docs):
        docs_str = ""
        for doc in docs:
            docs_str += doc.replace("\n"," ")
            docs_str += "\n\n" 
        docs_str = docs_str.strip() 

        prompt = TEMPLATE.format(characteristic=characteristics[args.char_idx], type=types[args.type_idx], style=styles[args.style_idx], answer_length=answer_lengths[args.length_idx], docs=docs_str)

        return prompt

    if args.use_existing_prompts:
        output_dir = args.given_prompt_dir.replace("/prompts", "")
    else:
        if args.straightforward_prompt:
            output_dir = f"{args.output_dir}/prompt_{args.prompt_num}_numdocs_{args.num_docs_to_prompt_with}"
        elif args.template_prompt:
            output_dir = f"{args.output_dir}/template_{args.char_idx}{args.type_idx}{args.style_idx}{args.length_idx}_numdocs_{args.num_docs_to_prompt_with}"

    # ensure output dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # access source data
    if args.use_train:
        all_files = [
            f
            for f in os.listdir(args.input_dir)
            if f.endswith("train.pt")
        ]
        num_rows_to_get_overall = args.num_rows_to_get_overall
    elif args.use_valid:
        all_files = [
            f
            for f in os.listdir(args.input_dir)
            if f.endswith("valid.pt")
        ]
        num_rows_to_get_overall = args.num_rows_to_get_overall*0.1
    elif args.use_test: 
        all_files = [
            f
            for f in os.listdir(args.input_dir)
            if f.endswith("test.pt")
        ]
    all_files = all_files[::-1]
    count = 0

    # consider each data split (train/valid/test)
    for file_idx, file_name in enumerate(tqdm(all_files)):
        print(file_name) 
        num_prompts=0

        ###### ACCESS DOCUMENT CLUSTERS
        # read the document clusters into dataframe
        # data_path = os.path.join(args.input_dir, file_name.replace('.pt', '.csv'))
        # clusters_df = pd.read_csv(data_path, index_col=None, converters={"articles": literal_eval, "cluster_id": literal_eval})
        # print("Total cluster count: ", len(clusters_df))

        
        data_path = os.path.join(args.input_dir, file_name)
        with open(data_path, "rb") as f:
            data_list = torch.load(f)

        for_df = {"articles": [], "cluster_id": []}
        for i, cluster in enumerate(data_list):
            for_df["articles"].append(cluster)
            for_df["cluster_id"].append(i)

        clusters_df = pd.DataFrame(for_df)

        # shuffle so it's randomized order (esp. in terms of cluster IDs)
        clusters_df = clusters_df.sample(n=len(clusters_df), random_state=42)

        # select desired number of rows to use for prompt generation
        multiplier = args.cluster_start_multiplier
        start_idx = (multiplier-1) * args.num_prompts_to_generate
        end_idx = multiplier * args.num_prompts_to_generate
        print("START AND END CLUSTER INDEX", start_idx, end_idx)
        selected_clusters_df = clusters_df.iloc[start_idx:end_idx]
        print("DIMENSIONS", selected_clusters_df.shape)

        ###### CREATE INSTRUCTION GENERATION PROMPTS FOR EACH CLUSTER
        all_prompts = []
        instrxn_cluster_ids = []
        source_docs = []
        
        prompt_dir = os.path.join(output_dir, "prompts") 
        if not os.path.exists(prompt_dir): 
            os.makedirs(prompt_dir)
        docs_dir = os.path.join(output_dir, "source_docs")
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)
        cluster_id_dir = os.path.join(output_dir, "cluster_ids")
        if not os.path.exists(cluster_id_dir):
            os.makedirs(cluster_id_dir)
    
        # if not using existing prompts
        if not args.use_existing_prompts:

            # store selected source content to lists for later use
            all_source_docs = selected_clusters_df['articles'].tolist()
            all_cluster_ids = selected_clusters_df['cluster_id'].tolist()
            file_idx = 0

            # generate the prompts according to pre-defined templates
            for cluster, cluster_id in zip(all_source_docs, all_cluster_ids):

                # randomize doc order
                random.shuffle(cluster)
                docs = [doc.replace("\n", " ") for doc in cluster]

                if args.num_docs_to_prompt_with!=-1:
                    docs = docs[:max(args.num_docs_to_prompt_with, len(docs))]
                
                if args.straightforward_prompt:
                    prompt = get_general_prompt(args.prompt_num, docs)
                elif args.template_prompt:
                    prompt = get_template_prompt(docs)

                # save current slice of prompt generations & metadata
                source_docs.append(cluster)
                all_prompts.append(prompt)
                instrxn_cluster_ids.append(cluster_id)
                num_prompts += 1

                # save current prompts and metadata to file if count=10
                if num_prompts > 9:
                    prompt_file_name = file_name[:-2]+"%d.pt"%(file_idx)
                    torch.save(all_prompts, os.path.join(prompt_dir, prompt_file_name))
                    print("Finished saving prompt slice %d at %s"%(file_idx, prompt_dir+"/"+prompt_file_name))

                    cluster_id_file_name = file_name[:-2]+"%d.pt"%(file_idx)
                    torch.save(instrxn_cluster_ids, os.path.join(cluster_id_dir, cluster_id_file_name))
                    print("Finished saving cluster ids slice %d at %s"%(file_idx, cluster_id_dir+"/"+cluster_id_file_name))

                    source_docs_file_name = file_name[:-2]+"%d.pt"%(file_idx)
                    torch.save(source_docs, os.path.join(docs_dir, source_docs_file_name))
                    print("Finished saving source documents slice %d at %s"%(file_idx, docs_dir+"/"+source_docs_file_name))

                    print("Just-saved number of prompts:",num_prompts)
                    num_prompts = 0
                    source_docs = []
                    all_prompts = []
                    instrxn_cluster_ids = []
                    file_idx += 1
            
            # save any remaining prompt generations & metadata if not already done 
            torch.save(all_prompts, os.path.join(prompt_dir, file_name[:-2]+"%d.pt"%(file_idx)))
            print("Finished saving last few prompts at %s"%(prompt_dir+"/"+file_name[:-2]+"%d.pt"%(file_idx)))

            torch.save(instrxn_cluster_ids, os.path.join(cluster_id_dir, file_name[:-2]+"%d.pt"%(file_idx)))
            print("Finished saving last few cluster_ids at %s"%(cluster_id_dir+"/"+file_name[:-2]+"%d.pt"%(file_idx)))

            torch.save(source_docs, os.path.join(docs_dir, file_name[:-2]+"%d.pt"%(file_idx)))
            print("Finished saving last few source documents at %s"%(docs_dir+"/"+file_name[:-2]+"%d.pt"%(file_idx)))

            print("Just-saved number of prompts:", num_prompts)
        
    ######################################################
    ######## Use prompts to generate instructions ########
    ######################################################
    if args.only_generate_prompts==False:

        # if using already-generated prompts, use correct prompt directory
        if args.use_existing_prompts:
            prompt_dir = args.given_prompt_dir

        # ensure output directory for generated instructions exists
        instrxn_dir = os.path.join(output_dir, "instructions")
        if not os.path.exists(instrxn_dir):
            os.makedirs(instrxn_dir)
    
        # read in all the prompt files
        all_files = sorted([
            f
            for f in os.listdir(prompt_dir)
            if f.endswith(".pt")
        ])

        # consider each prompt file 
        num_slices_done = 0
        for file_idx, pt_file in enumerate(tqdm(all_files)):

            # control generation via number of prompt slices(files) done
            # exit if all done
            if args.num_slices_to_do!= -1 and args.num_slices_done > args.num_slices_to_do:
                break 
                                
            # Ppss if not yet at desired prompt file start index 
            if args.start_instr_file_idx!=-1 and file_idx < args.start_instr_file_idx:
                continue
            
            # generate if no start index specified or if we've reached the start index
            elif (args.start_instr_file_idx==-1) or (args.start_instr_file_idx!=-1 and file_idx >= args.start_instr_file_idx): 

                # write current filename to log (to assist pause/resume)
                print("#"*20)
                print("On slice", file_idx)
                f = open(os.path.join(output_dir,"num_instr_slices_generated_so_far.txt"), "w")
                f.write("\nCurerntly generating instructions for prompt file: "+str(pt_file))
                f.write("\nCorresponding file_idx: "+str(file_idx))
                f.close()

                # prompt GPT
                prompt_slice = torch.load(os.path.join(prompt_dir, pt_file))
                instructions = [gpt_generate(model=args.model, prompt=prompt).choices[0].message.content.strip() for prompt in prompt_slice]

                instr_file_name = pt_file

                torch.save(instructions, os.path.join(instrxn_dir, instr_file_name))
                print("Finished saving generated instructions part %d at %s"%(file_idx, instrxn_dir+"/"+instr_file_name))

                num_slices_done += 1

        print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Directory control
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./0_source_data_preparation/base_articles",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./1_mdcure_generation/generations_style_specific",
    )

    # Prompt usage control
    parser.add_argument(
        '--use_existing_prompts', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--only_generate_prompts', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--given_prompt_dir",
        type=str,
        default=None,
    )

    # Generation control
    parser.add_argument(
        "--start_instr_file_idx",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--cluster_start_multiplier",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--num_prompts_to_generate",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--num_slices_to_do",
        type=int,
        default=-1,
    )

    # Source docs control
    parser.add_argument(
        '--use_train', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_valid', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_test', 
        action='store_true',
        default=False,
    )

    # Prompting control
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
    )
    parser.add_argument(
        "--num_docs_to_prompt_with",
        type=int,
        default=2,
    )
    parser.add_argument(
        '--straightforward_prompt', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--prompt_num",
        type=int,
        default=None,
    )
    parser.add_argument(
        '--template_prompt', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--char_idx",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--type_idx",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--style_idx",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--length_idx",
        type=int,
        default=None,
    )
    parser.add_argument(
        '--num_rows_to_get_overall', 
        type=int,
        default=300000,
    )
    args = parser.parse_args()
    print(args)
    
    generate_and_save_instructions(args)


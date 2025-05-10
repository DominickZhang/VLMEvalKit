from vlmeval.smp import *
import os
import sys
from vlmeval.api.base import BaseAPI
import math
from vlmeval.dataset import DATASET_TYPE
from vlmeval.dataset import img_root_map
from io import BytesIO
import pandas as pd
import requests
import json
import base64
import time

os.environ['http_proxy'] = 'http://star-proxy.oa.com:3128'
os.environ['https_proxy'] = 'http://star-proxy.oa.com:3128'

def chat(base64_image, prompt, api_key, model_marker, model_name, sytem_prompt=""):
    # base64_image,image_format = base64_encode_image(img_path)
    # print("文件类型：",image_format.lower())
    # img_txt_prompt = f"data:image/jpeg;base64,{base64_image}"
    # img_txt_prompt = f"data:image/{image_format.lower()};base64,{base64_image}"
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "value": prompt,
                },
                {
                    "type": "image_url",
                    "value": base64_image,
                },
            ],
        },
    ]

    # {api_key}为token，{request_id}可用hash生成，超参写入{params}
    json_data = {
        "bid": "open_api_test",
        "server": "open_api",
        "services": [],
        "request_id": "1234",
        "session_id": "12345",  
        # "api_key": "90237a2c-0b99-4d4e-8e78-b3473ad3e13c",
        # "api_key": "2dbc9edb-ecf8-4860-85b4-0af788868221",
        # "api_key": "081aaf2c-e5b4-4718-bf55-02acc151f65f",
        "api_key": api_key,
        # "model_marker":"api_doubao_Doubao-vision-pro-32k-241028",
        # "model_marker": "api_openai_chatgpt-4o-latest",
        "model_marker": model_marker,
        "system": "", # 模型人设
        "params":{},
        "timeout": 300, # 超时时间,单位秒
        # "model_name": "api_openai_chatgpt-4o-latest",
        "model_name": model_name,
        # "model_name":"Doubao-pro-128k",
        # "model_name":"Doubao-pro-32k",
        "messages": messages
    }

    url= "http://trpc-utools-prod.turbotke.production.polaris:8009/"
    
    res = requests.post(url=url, json=json_data, proxies={"http": None, "https": None})
    # print(res.json()['answer'][0]['value'])
    if res.status_code == 200 and res is not None:
        # print("res.json()",res.json())
        # print(res.json())
        return(res.json()['answer'][0]['value'])
    else:
        return ""

class HunYuanAPIWrapper(BaseAPI):

    is_api: bool = True
    _apiVersion = '2025-05-11'
    _service = 'hunyuan_api'

    def __init__(self,
                 model: str = 'gpt-4o-latest',
                 retry: int = 5,
                 wait: int = 5,
                 secret_key: str = None,
                 secret_id: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 60,
                 api_base: str = 'hunyuan.tencentcloudapi.com',
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.temperature = temperature

        warnings.warn('You may need to set the env variable HUNYUAN_SECRET_ID & HUNYUAN_SECRET_KEY to use Hunyuan. ')

        ## HUNYUAN_SECRET_KEY 对应api_key
        secret_key = os.environ.get('HUNYUAN_SECRET_KEY', secret_key)
        assert secret_key is not None, 'Please set the environment variable HUNYUAN_SECRET_KEY. '
        # HUNYUAN_SECRET_ID 对应model_marker
        secret_id = os.environ.get('HUNYUAN_SECRET_ID', secret_id)
        assert secret_id is not None, 'Please set the environment variable HUNYUAN_SECRET_ID. '
        ## HUNYUAN_MODEL_NAME 对应 model_name
        model_name = os.environ.get('HUNYUAN_MODEL_NAME', model_name)
        assert secret_key is not None, 'Please set the environment variable HUNYUAN_MODEL_NAME. '

        ## 这个没有用到
        self.model = model

        # self.endpoint = api_base
        self.model_name = model_name
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.model_marker = secret_id
        self.api_key = secret_key
        self.timeout = timeout

        '''
        try:
            from tencentcloud.common import credential
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.common.profile.http_profile import HttpProfile
            from tencentcloud.hunyuan.v20230901 import hunyuan_client
        except ImportError as err:
            self.logger.critical('Please install tencentcloud-sdk-python to use Hunyuan API. ')
            raise err
        '''

        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        '''
        cred = credential.Credential(self.secret_id, self.secret_key)
        httpProfile = HttpProfile(reqTimeout=300)
        httpProfile.endpoint = self.endpoint
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        self.client = hunyuan_client.HunyuanClient(cred, '', clientProfile)
        '''
        self.logger.info(
            f'Model Name: {self.model_name}; API Secret ID: {self.secret_id}; API Secret Key: {self.secret_key}'
        )

    def dump_image(self, line, dataset):
        """Dump the image(s) of the input line to the corresponding dataset folder.

        Args:
            line (line of pd.DataFrame): The raw input line.
            dataset (str): The name of the dataset.

        Returns:
            str | list[str]: The paths of the dumped images.
        """
        ROOT = LMUDataRoot()
        assert isinstance(dataset, str)

        img_root = os.path.join(ROOT, 'images', img_root_map(dataset) if dataset in img_root_map(dataset) else dataset)
        os.makedirs(img_root, exist_ok=True)
        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                assert 'image_path' in line
                for img, im_name in zip(line['image'], line['image_path']):
                    path = osp.join(img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)
            else:
                tgt_path = osp.join(img_root, f"{line['index']}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])

        return tgt_path

    def use_custom_prompt(self, dataset_name):
        if DATASET_TYPE(dataset_name) == 'MCQ':
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)

        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Answer with the option letter from the given choices directly.'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(Type='text', Text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img)
                    img_struct = dict(Url=f'data:image/jpeg;base64,{b64}')
                    content_list.append(dict(Type='image_url', ImageUrl=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(Type='text', Text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        # if self.system_prompt is not None:
        #     input_msgs.append(dict(Role='system', Content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs

        # messages=[
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "text",
        #                 "value": prompt,
        #             },
        #             {
        #                 "type": "image_url",
        #                 "value": base64_image,
        #             },
        #         ],
        #     },
        # ]

        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(Role=item['role'], Contents=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(Role='user', Contents=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        # from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
        # from tencentcloud.hunyuan.v20230901 import models

        messages = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)

        # {api_key}为token，{request_id}可用hash生成，超参写入{params}
        json_data = {
            "bid": "open_api_test",
            "server": "open_api",
            "services": [],
            "request_id": "1234",
            "session_id": "12345",  
            # "api_key": "90237a2c-0b99-4d4e-8e78-b3473ad3e13c",
            # "api_key": "2dbc9edb-ecf8-4860-85b4-0af788868221",
            # "api_key": "081aaf2c-e5b4-4718-bf55-02acc151f65f",
            "api_key": self.api_key,
            # "model_marker":"api_doubao_Doubao-vision-pro-32k-241028",
            # "model_marker": "api_openai_chatgpt-4o-latest",
            "model_marker": self.model_marker,
            "system": "", # 模型人设
            "params":{"temperature": temperature},
            "timeout": 300, # 超时时间,单位秒
            # "model_name": "api_openai_chatgpt-4o-latest",
            "model_name": self.model_name,
            # "model_name":"Doubao-pro-128k",
            # "model_name":"Doubao-pro-32k",
            "messages": messages
        }

        url= "http://trpc-utools-prod.turbotke.production.polaris:8009/"
        
        res = requests.post(url=url, json=json_data, proxies={"http": None, "https": None})
        # print(res.json()['answer'][0]['value'])
        if res.status_code == 200 and res is not None:
            # print("res.json()",res.json())
            # print(res.json())
            return 0, (res.json()['answer'][0]['value']), res.json()
        else:
            return -1, res, None

        '''
        payload = dict(
            Model=self.model,
            Messages=input_msgs,
            Temperature=temperature,
            TopK=1,
            **kwargs)

        try:
            req = models.ChatCompletionsRequest()
            req.from_json_string(json.dumps(payload))
            resp = self.client.ChatCompletions(req)
            resp = json.loads(resp.to_json_string())
            answer = resp['Choices'][0]['Message']['Content']
            return 0, answer, resp
        except TencentCloudSDKException as e:
            self.logger.error(f'Got error code: {e.get_code()}')
            if e.get_code() == 'ClientNetworkError':
                return -1, self.fail_msg + e.get_code(), None
            elif e.get_code() in ['InternalError', 'ServerNetworkError']:
                return -1, self.fail_msg + e.get_code(), None
            elif e.get_code() in ['LimitExceeded']:
                return -1, self.fail_msg + e.get_code(), None
            else:
                return -1, self.fail_msg + str(e), None
        '''


class HunYuanAPI(HunYuanAPIWrapper):

    def generate(self, message, dataset=None):
        return super(HunYuanAPI, self).generate(message)

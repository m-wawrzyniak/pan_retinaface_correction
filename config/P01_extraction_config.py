# PROJECT
from sklearn.utils import TargetTags
from torchgen.utils import Target

ROOT_LOCAL = "/home/mateusz-wawrzyniak/PycharmProjects/pan_retinaface_correction"

ROOT = "/media/mateusz-wawrzyniak/Extreme SSD/IP_PAN/retinaface_secondary_classification"
PROJECT_STRUCT = f"{ROOT}/config/project_dir_structure.json"

DATASET_NAME = "dataset_full_v00"
CLASSIFIER_NAME = "class_full_v00"

# EXTRACTION

# locals
"""
TIMESERIES_DATA = "/home/mateusz-wawrzyniak/Desktop/IP_PAN_Videos/Timeseries Data + Scene Video/"
SECTIONS_CSV = "/home/mateusz-wawrzyniak/Desktop/IP_PAN_Videos/Sit&Face_FACE-MAPPER_Faces_Manipulative/sections.csv"
FACE_MAPPER_DIR = "/home/mateusz-wawrzyniak/Desktop/IP_PAN_Videos/Sit&Face_FACE-MAPPER_Faces_Manipulative/"
"""
# externals
TIMESERIES_DATA = "/media/mateusz-wawrzyniak/Extreme SSD/IP_PAN/Timeseries Data + Scene Video/"
SECTIONS_CSV = f"/media/mateusz-wawrzyniak/Extreme SSD/IP_PAN/Sit&Face_FACE-MAPPER_Faces_Manipulative/sections.csv"
FACE_MAPPER_DIR = "/media/mateusz-wawrzyniak/Extreme SSD/IP_PAN/Sit&Face_FACE-MAPPER_Faces_Manipulative/"

REC_SUBSET = ['dd674314-3535-432e-be52-c6f2a6f51714', 'ddd76a84-0e70-4198-8863-43996ccc1844', '6e1f974f-edcc-48b3-a2ce-40f9d996e9f6',
            '94138e31-6f7b-4c65-8e6f-8b3ea1b9aea4', '11d92ae2-4d0b-45fe-b75c-fa217c1d4e62', 'e3ae086e-8d4d-4964-98b6-56b751753496',
            '7140215c-3cdc-40ae-85d2-1e1c7b1cac29', '965e52b1-985e-428d-bb59-9d0e280f81bf', '26950250-561c-40a9-83a6-73df88bc32b4',
            'a38fdd5d-3cae-4453-b9b5-4ddd6f178aa2', '58ab60b8-28ee-42c3-a236-99153c860857', '7e3e6fc7-6e99-4a8d-bcc3-58192c016a5e',
            'd4848c49-293d-4248-b3ef-726514db4049', '628f7e62-164d-4bee-8e2c-60df66d47ed8', '6e2abfde-326f-48e1-a1c2-0166314c4d59',
            '13dc41fe-dd1f-43bf-b160-865b5f94fb27', 'b1559499-2881-40b0-8041-be04d23aeff4', '550f31a7-fc1f-4753-be92-83de953a537b',
            '2707aac6-5d63-4699-9646-a0abdf0c6621', '07a74876-f775-460c-b727-6723d0c2a95c', '52a54554-a1d8-4dfb-867e-d46f099a3bcd',
            '8c03a1f3-4627-4e0a-b90f-7a52d9faee3d', '6943ea9b-a2bc-4e40-836b-fdacbba22ad5', '2a351890-f23d-400a-abd0-7003793f7648',
            'be261083-feef-4627-a71b-54708ccb3b88', 'fb68bd46-423c-4c5c-9211-4a8ec7c3f068', '5375387b-6cf7-492e-a866-be923395c692',
            'b409259e-d2a6-420d-b3fa-58d92b2b74f6', 'b2a83ebc-1d27-4688-81de-8b2c1ad612a9', '1f74e826-4258-46cf-96ee-8776f40f805c',
            'ad581215-0012-430a-8d26-9675b83f6180', '8ae9fdcc-adbf-4740-acaf-6ac464501075', '747f2f18-8f83-406d-ae7b-ccab081a47ac',
            'c825e18a-7fb1-4e7e-b364-4e30938c85ba', 'f794f422-33ae-41c1-b59d-254ce18fb935', '0bc94afe-8ec4-43a5-807f-b26b9a835b13',
            '9829711b-aa8e-4330-be9c-c87d6c51087d', 'a204b6c6-db27-47cb-83aa-ac32b1eae9a3', 'c4867348-5c52-4acd-81c9-e48bcb51efdc',
            'cdeb4d00-f2ff-4282-abe0-bb9963adccc5', '1237a7c2-ebec-46d5-9ec9-3a2337712836', 'd7beb427-64bf-4aaa-b18b-205a95dca631',
            'd06d7ada-bbf9-4497-9dbd-24a9df672986', 'c8e88b5e-80ca-4b64-a324-5fdd6ad2cdc7', 'a5113ba6-3aea-411f-9703-ca3e5a057940',
            'fb8586c7-5350-4c9e-84d1-7d195d2d71e3', 'cebd1bec-72f0-4084-9747-31916ffab40a', '141eadd9-36dc-4c0b-9ad5-2c61ea8db603',
            '1764d820-c8cb-4e93-a6ba-95f81c271a33', '7d5f9c94-3841-405c-a882-b9a1ba95aaba', 'b9572652-7a81-478d-9474-ce3fda17b342',
            'ffcc7d92-551e-4562-8ce0-fd25c38623e3', 'ae4795fe-1a87-429d-9490-49b414b50f64', 'c4b6f4b3-1aaa-4bb9-b06b-f50aea17db9f',
            '9732fece-9eaf-4ccf-9ff4-d20590e80940', '9661f15c-647b-4de3-922d-bdedbfa9e2e4', '8be83c7e-ebb0-4b48-ae73-b441ea9c84a4',
            '3fd3b5a8-5563-4304-8533-c385130beb9e', '42ba038b-dc28-4317-98e0-59aede116741', 'b53417e9-f382-4740-9248-365d6faee68d',
            '9128c5ca-2b4e-4129-8bc2-8e71e6a3f5f6', '429bd4c2-acf7-486d-8ae7-11268503f50c', '4274e8e9-0f30-4663-b862-81c8f4f90fe7',
            '17713109-b3fc-4270-9d11-641f262c833f', '500211d7-9dea-4907-8067-249a686ebaf7',
            '0b5cefee-d2c4-41b3-a7a1-80085cdb87d0', 'd2bb2a5b-7e2d-40bf-848c-020429f3f16f', '948ab8ba-ee02-4671-8517-4c920a98eb46',
            'b87b3d74-f568-4954-a70b-ca28c69dee8c', '70ca44e7-a053-4d79-8b80-35157d35c046', '419b1cc6-d47e-4156-af08-a184a82a156a',
            '808af629-6d29-48d7-b1fc-cc671945eede', '675a8889-7021-4f71-8710-32eebdb9c1df', 'cc219207-41e6-4d5c-aa07-01ff185d07c5',
            '838fb1ef-9267-470f-970b-4ce30a09281e', '783d932c-f7bc-4d6b-b625-501055d7e82e', 'ab9958bf-4dff-4618-97c8-b5c23c72f1b0',
            '696d944e-f2a9-4274-95d8-7df9e35e698d', '1083cf2b-b67a-4362-a7c1-5b911fc0108c', '0eeed66d-6365-43ce-b99c-4b57b5788fe8',
            '14b51289-476a-4d4d-a0cf-5175b36d66ad', '5530dc89-b661-4f34-a50b-1ea3e4a35e2d', '5868dd07-ad2e-40f9-9a29-b321cddf6e3f',
            'dfa33c11-1bfa-4aa8-9e65-df4b38afd256', '75806b5d-21f5-4df3-b2e9-2e1ac77cdbe0', 'ee95e947-5f52-4a43-a463-949605be4e0a',
            'f6909c5f-6564-445b-98c1-ef71a7de4935', 'fbfd7032-cab2-4fb6-a268-ec4b86020a6d', 'd1215b83-70a7-4196-994e-eb9622e9faa2',
            '98800b2a-d21b-4423-a118-9aea3a6421cd', '847bf97d-c5db-438a-b428-0ae80a1e3161', 'b45534e6-4374-4681-9c09-4ed88015dfa2',
            '07373e39-c80a-4751-bc39-6be1c74b977c', 'bf3d83f3-b5de-48d7-82e7-bed1ee3c49ca', '45ff84be-38d3-45e2-9f81-8fef1ad98a89',
            '8f355b69-e293-4da0-a1a2-6d711ca56d54', '8b0f9c84-d82a-4d55-83a0-c4082c52f221', '4566db4d-7098-4f6a-adf2-ef293edc149c',
            '0da7704a-11a3-42a4-bcf4-a7896c1e1b65', 'b3bb1d0c-421c-4f17-83aa-58f4bbfd58de', '5c3ba006-754f-4902-94d5-a0235200d8b5',
            '6db584c1-ee5c-4770-894e-f9ec18a653fb', 'e1dfc12f-1984-4f26-b919-b4042df50eba', '7be92857-0f1c-478d-86de-1280adcd4689',
            'd4bebafd-ce12-4de3-8afc-d0d7c77ce2ef', '2bf892c8-de71-4b74-9010-d6832b35bab9', '7f755a0f-3a9c-48fd-ac0c-2cc691c6a9a9',
            '80f18cec-1762-43ca-ac78-3f7df0dc7159', 'd465c077-1431-4cb4-aafc-59ea7e003c94', '3981c3b0-09c5-4035-9028-c6298660e202',
            'f33229d8-4552-4f03-88f7-bbc68d7e11f2', 'f9c81e2d-533b-4445-bfc6-2e3a0f70663a', 'b7df6039-7b3d-4991-9086-65e6e17a5d9d',
            '556c641c-4c51-4083-b16c-0ebcd7c19d0d', 'dfb7459c-e21d-41d1-823c-65fc650f851f', '338d15e1-441d-4097-ad0e-da51d30b35ed',
            '8b79979e-7e0b-4fcb-ba02-ae07cc6ac18a', 'e10812f1-3b15-4773-a558-ce10d65835cd', '01cb98ef-cc66-4d2f-9878-5a17216cd116',
            'a40d5a0e-cd29-40d9-8045-06a9f67c0072', '80045ad1-d139-49fa-ac17-4c92d7628f59', '3a031d8a-80ec-4fd2-af5e-c8dd06b31e6d',
            '32fada6a-02ac-4a2a-831b-ae46c22c57b1', '401b2a79-2406-4c46-8079-4f75c019815a', 'bbfad8c5-316a-4d7b-9a57-421185a15d11',
            '00491ba9-6987-4433-aa04-890cdd1da27d', 'f8c627fc-4bdd-4e1f-9d60-ef03ad80184e', 'dcd95915-e5b0-4220-99b2-19c883d41d33',
            '76eb2fea-912c-4557-97a9-560c14559bfe', '0be8b47a-4cee-409f-a641-d16e390f6303', 'a93ce8d5-e813-412f-b28e-6ed307f59e3b',
            'c577980a-7103-42f6-92bf-6d9c07c77094', 'ac8f0ac1-8893-4a3f-a1aa-aee51ed653f4']

REC_SUBSET_SMALL = [
    "dcd95915-e5b0-4220-99b2-19c883d41d33"
]

REC_SUBSET_MEDIUM = [
    "dcd95915-e5b0-4220-99b2-19c883d41d33",
    "a93ce8d5-e813-412f-b28e-6ed307f59e3b",
    "ac8f0ac1-8893-4a3f-a1aa-aee51ed653f4",
    "0be8b47a-4cee-409f-a641-d16e390f6303",
    "b3bb1d0c-421c-4f17-83aa-58f4bbfd58de",
    "1764d820-c8cb-4e93-a6ba-95f81c271a33",
    "3a031d8a-80ec-4fd2-af5e-c8dd06b31e6d",
    "d1215b83-70a7-4196-994e-eb9622e9faa2",
    "e1dfc12f-1984-4f26-b919-b4042df50eba",
    "f8c627fc-4bdd-4e1f-9d60-ef03ad80184e",
    "b9572652-7a81-478d-9474-ce3fda17b342",
    "76eb2fea-912c-4557-97a9-560c14559bfe"
]

REC_FULL = ['dd674314-3535-432e-be52-c6f2a6f51714', 'ddd76a84-0e70-4198-8863-43996ccc1844', '6e1f974f-edcc-48b3-a2ce-40f9d996e9f6',
            '94138e31-6f7b-4c65-8e6f-8b3ea1b9aea4', '11d92ae2-4d0b-45fe-b75c-fa217c1d4e62', 'e3ae086e-8d4d-4964-98b6-56b751753496',
            '7140215c-3cdc-40ae-85d2-1e1c7b1cac29', '965e52b1-985e-428d-bb59-9d0e280f81bf', '26950250-561c-40a9-83a6-73df88bc32b4',
            'a38fdd5d-3cae-4453-b9b5-4ddd6f178aa2', '58ab60b8-28ee-42c3-a236-99153c860857', '7e3e6fc7-6e99-4a8d-bcc3-58192c016a5e',
            'd4848c49-293d-4248-b3ef-726514db4049', '628f7e62-164d-4bee-8e2c-60df66d47ed8', '6e2abfde-326f-48e1-a1c2-0166314c4d59',
            '13dc41fe-dd1f-43bf-b160-865b5f94fb27', 'b1559499-2881-40b0-8041-be04d23aeff4', '550f31a7-fc1f-4753-be92-83de953a537b',
            '2707aac6-5d63-4699-9646-a0abdf0c6621', '07a74876-f775-460c-b727-6723d0c2a95c', '52a54554-a1d8-4dfb-867e-d46f099a3bcd',
            '8c03a1f3-4627-4e0a-b90f-7a52d9faee3d', '6943ea9b-a2bc-4e40-836b-fdacbba22ad5', '2a351890-f23d-400a-abd0-7003793f7648',
            'be261083-feef-4627-a71b-54708ccb3b88', 'fb68bd46-423c-4c5c-9211-4a8ec7c3f068', '5375387b-6cf7-492e-a866-be923395c692',
            'b409259e-d2a6-420d-b3fa-58d92b2b74f6', 'b2a83ebc-1d27-4688-81de-8b2c1ad612a9', '1f74e826-4258-46cf-96ee-8776f40f805c',
            'ad581215-0012-430a-8d26-9675b83f6180', '8ae9fdcc-adbf-4740-acaf-6ac464501075', '747f2f18-8f83-406d-ae7b-ccab081a47ac',
            'c825e18a-7fb1-4e7e-b364-4e30938c85ba', 'f794f422-33ae-41c1-b59d-254ce18fb935', '0bc94afe-8ec4-43a5-807f-b26b9a835b13',
            '9829711b-aa8e-4330-be9c-c87d6c51087d', 'a204b6c6-db27-47cb-83aa-ac32b1eae9a3', 'c4867348-5c52-4acd-81c9-e48bcb51efdc',
            'cdeb4d00-f2ff-4282-abe0-bb9963adccc5', '1237a7c2-ebec-46d5-9ec9-3a2337712836', 'd7beb427-64bf-4aaa-b18b-205a95dca631',
            'd06d7ada-bbf9-4497-9dbd-24a9df672986', 'c8e88b5e-80ca-4b64-a324-5fdd6ad2cdc7', 'a5113ba6-3aea-411f-9703-ca3e5a057940',
            'fb8586c7-5350-4c9e-84d1-7d195d2d71e3', 'cebd1bec-72f0-4084-9747-31916ffab40a', '141eadd9-36dc-4c0b-9ad5-2c61ea8db603',
            '1764d820-c8cb-4e93-a6ba-95f81c271a33', '7d5f9c94-3841-405c-a882-b9a1ba95aaba', 'b9572652-7a81-478d-9474-ce3fda17b342',
            'ffcc7d92-551e-4562-8ce0-fd25c38623e3', 'ae4795fe-1a87-429d-9490-49b414b50f64', 'c4b6f4b3-1aaa-4bb9-b06b-f50aea17db9f',
            '9732fece-9eaf-4ccf-9ff4-d20590e80940', '9661f15c-647b-4de3-922d-bdedbfa9e2e4', '8be83c7e-ebb0-4b48-ae73-b441ea9c84a4',
            '3fd3b5a8-5563-4304-8533-c385130beb9e', '42ba038b-dc28-4317-98e0-59aede116741', 'b53417e9-f382-4740-9248-365d6faee68d',
            '9128c5ca-2b4e-4129-8bc2-8e71e6a3f5f6', '429bd4c2-acf7-486d-8ae7-11268503f50c', '4274e8e9-0f30-4663-b862-81c8f4f90fe7',
            '17713109-b3fc-4270-9d11-641f262c833f', '500211d7-9dea-4907-8067-249a686ebaf7',
            '0b5cefee-d2c4-41b3-a7a1-80085cdb87d0', 'd2bb2a5b-7e2d-40bf-848c-020429f3f16f', '948ab8ba-ee02-4671-8517-4c920a98eb46',
            'b87b3d74-f568-4954-a70b-ca28c69dee8c', '70ca44e7-a053-4d79-8b80-35157d35c046', '419b1cc6-d47e-4156-af08-a184a82a156a',
            '808af629-6d29-48d7-b1fc-cc671945eede', '675a8889-7021-4f71-8710-32eebdb9c1df', 'cc219207-41e6-4d5c-aa07-01ff185d07c5',
            '838fb1ef-9267-470f-970b-4ce30a09281e', '783d932c-f7bc-4d6b-b625-501055d7e82e', 'ab9958bf-4dff-4618-97c8-b5c23c72f1b0',
            '696d944e-f2a9-4274-95d8-7df9e35e698d', '1083cf2b-b67a-4362-a7c1-5b911fc0108c', '0eeed66d-6365-43ce-b99c-4b57b5788fe8',
            '14b51289-476a-4d4d-a0cf-5175b36d66ad', '5530dc89-b661-4f34-a50b-1ea3e4a35e2d', '5868dd07-ad2e-40f9-9a29-b321cddf6e3f',
            'dfa33c11-1bfa-4aa8-9e65-df4b38afd256', '75806b5d-21f5-4df3-b2e9-2e1ac77cdbe0', 'ee95e947-5f52-4a43-a463-949605be4e0a',
            'f6909c5f-6564-445b-98c1-ef71a7de4935', 'fbfd7032-cab2-4fb6-a268-ec4b86020a6d', 'd1215b83-70a7-4196-994e-eb9622e9faa2',
            '98800b2a-d21b-4423-a118-9aea3a6421cd', '847bf97d-c5db-438a-b428-0ae80a1e3161', 'b45534e6-4374-4681-9c09-4ed88015dfa2',
            '07373e39-c80a-4751-bc39-6be1c74b977c', 'bf3d83f3-b5de-48d7-82e7-bed1ee3c49ca', '45ff84be-38d3-45e2-9f81-8fef1ad98a89',
            '8f355b69-e293-4da0-a1a2-6d711ca56d54', '8b0f9c84-d82a-4d55-83a0-c4082c52f221', '4566db4d-7098-4f6a-adf2-ef293edc149c',
            '0da7704a-11a3-42a4-bcf4-a7896c1e1b65', 'b3bb1d0c-421c-4f17-83aa-58f4bbfd58de', '5c3ba006-754f-4902-94d5-a0235200d8b5',
            '6db584c1-ee5c-4770-894e-f9ec18a653fb', 'e1dfc12f-1984-4f26-b919-b4042df50eba', '7be92857-0f1c-478d-86de-1280adcd4689',
            'd4bebafd-ce12-4de3-8afc-d0d7c77ce2ef', '2bf892c8-de71-4b74-9010-d6832b35bab9', '7f755a0f-3a9c-48fd-ac0c-2cc691c6a9a9',
            '80f18cec-1762-43ca-ac78-3f7df0dc7159', 'd465c077-1431-4cb4-aafc-59ea7e003c94', '3981c3b0-09c5-4035-9028-c6298660e202',
            'f33229d8-4552-4f03-88f7-bbc68d7e11f2', 'f9c81e2d-533b-4445-bfc6-2e3a0f70663a', 'b7df6039-7b3d-4991-9086-65e6e17a5d9d',
            '556c641c-4c51-4083-b16c-0ebcd7c19d0d', 'dfb7459c-e21d-41d1-823c-65fc650f851f', '338d15e1-441d-4097-ad0e-da51d30b35ed',
            '8b79979e-7e0b-4fcb-ba02-ae07cc6ac18a', 'e10812f1-3b15-4773-a558-ce10d65835cd', '01cb98ef-cc66-4d2f-9878-5a17216cd116',
            'a40d5a0e-cd29-40d9-8045-06a9f67c0072', '80045ad1-d139-49fa-ac17-4c92d7628f59', '3a031d8a-80ec-4fd2-af5e-c8dd06b31e6d',
            '32fada6a-02ac-4a2a-831b-ae46c22c57b1', '401b2a79-2406-4c46-8079-4f75c019815a', 'bbfad8c5-316a-4d7b-9a57-421185a15d11',
            '00491ba9-6987-4433-aa04-890cdd1da27d', 'f8c627fc-4bdd-4e1f-9d60-ef03ad80184e', 'dcd95915-e5b0-4220-99b2-19c883d41d33',
            '76eb2fea-912c-4557-97a9-560c14559bfe', '0be8b47a-4cee-409f-a641-d16e390f6303', 'a93ce8d5-e813-412f-b28e-6ed307f59e3b',
            'c577980a-7103-42f6-92bf-6d9c07c77094', 'ac8f0ac1-8893-4a3f-a1aa-aee51ed653f4']

REC_NO_MANIPULATIVE = [
    '96ded5e5-47a1-483e-8a76-ce07b0b0e37c', '0a32f22b-85d9-4c3c-b1c7-1cdadc34befd', '71dc09fa-0a53-4208-b38b-881709dcb180',
    '0ef2fdec-fbb9-48e4-b4d6-1fd0ee387545', '1d78ac8a-71d2-4519-a32e-f40ff3d354c0', '120138b9-4cc9-4485-b3c8-a3363ecd3b26',
    'b75417e6-6d01-4569-8eac-a47aa6b1eb26', 'f882b4e7-e656-4e81-8655-b5c6546be2c9', 'adb061a9-6cab-45fa-abcd-dcd029e6f35b',
    '6d2dbc6e-3f8d-4114-b815-35f927f602a0', '92ab3a43-8071-4f67-bdd1-e4b7169be14a', '16f8414e-4f62-4023-a1b7-062c248a6ab3',
    'e705c649-f5b7-4c1d-b01d-4de5f80bc3c3', 'ba9d0136-32f3-466f-bcb1-06d1e78c06db', '42c5720d-219e-4acb-93e6-38ee03a05644',
    'a07b6334-b9d9-4d25-b7f2-a090bf91fedb', '386e434e-9a84-49ef-a6bd-137cf93e4da1', 'df53e0fb-38d2-4cff-90f3-fc0eac0e759a',
    '55a410ba-5782-4a79-a2e5-69e631ff39c8', 'f30a1023-93cd-40ed-82cb-f2af65ad94e0', 'ca3de9bd-22b9-49e6-9fdf-58dfc1d784d4',
    'feabc5bf-aaea-4a28-bf49-8c727c7cb135', 'c721f793-f6c1-4b5a-a036-f663204d211f', '1adc3e53-7234-4d0d-9dcb-b0337ac86bb1',
    'b728a61f-2296-4281-a2ec-ca72152d2e9c', '0ea4acbd-d971-4174-ab36-7464e3aa4af6', '22e786db-28f9-43ec-b206-ed7e323bde2a',
    '0135b361-1728-4d69-b3c8-ca50493c4a2d', 'b72a7a3a-8dae-4055-9f13-484fd6707497', '4b5de67d-0d59-43f0-aad5-715b672dfee0',
    '9c9d5657-eb5a-4f2b-8dd4-163303fb9060', '73938361-987a-4db3-b329-19639ea52374', '869725e3-79a9-420e-9747-4dd1010e9575',
    'a1a3b788-73dc-48a9-8341-96cd04dd9d20', 'd2918415-7f74-4f3d-823b-89cf88f9eaf9', '964c65dc-9104-450d-a563-c21c809b90cb',
    '2271dec0-5972-4cb3-9c70-c73913ee50e3', '1ed94617-4582-4e9f-be32-262072239b33', '4d805ccd-0a9d-41d8-a142-be6656b1b976',
    'c0dddfe7-6068-488d-b08f-53651211af75', '9ac46c68-183a-454e-9305-17c6978c8261', '9a54afa6-7a3d-49e8-bde8-4aacc5bbdf93',
]

# Extraction parameters
DEDUP_THRESHOLD = 0.2
N_CLUSTERS = 12
MIN_PER_CLUSTER = 2
TARGET_FRAMES_PER_REC = 120
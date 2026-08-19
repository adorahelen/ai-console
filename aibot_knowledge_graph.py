import os
import json
import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from tqdm import tqdm
import difflib
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher
import itertools


@dataclass
class KnowledgeNode:
    id: str
    type: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    name_ko: str = None


@dataclass
class KnowledgeRelation:
    source: str
    target: str
    type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileProcessingRecord:
    file_path: str
    file_hash: str
    processed_time: float
    doc_id: str
    nodes_created: List[str] = field(default_factory=list)
    relations_created: List[str] = field(default_factory=list)


def calculate_similarity_fast(name1: str, name2: str, threshold: float = 0.7) -> float:
    def clean_name(name):
        name = name.lower().strip()
        name = re.sub(r'[^\w\s가-힣]', '', name)
        return re.sub(r'\s+', ' ', name)
    
    clean1 = clean_name(name1)
    clean2 = clean_name(name2)
    
    if not clean1 or not clean2:
        return 0.0
    
    len_ratio = min(len(clean1), len(clean2)) / max(len(clean1), len(clean2))
    if len_ratio < 0.5:
        return 0.0
    
    similarity = SequenceMatcher(None, clean1, clean2).ratio()
    return similarity if similarity >= threshold else 0.0

def process_similarity_batch(args):
    concept_pairs, threshold = args
    results = []
    
    for concept1, concept2 in concept_pairs:
        try:
            similarity = calculate_similarity_fast(concept1.name, concept2.name, threshold)
            
            if similarity == 0.0 and concept1.name_ko and concept2.name_ko:
                similarity = calculate_similarity_fast(concept1.name_ko, concept2.name_ko, threshold)
            
            if similarity == 0.0:
                for alias1 in concept1.aliases:
                    for alias2 in concept2.aliases:
                        alias_sim = calculate_similarity_fast(alias1, alias2, threshold)
                        if alias_sim > similarity:
                            similarity = alias_sim
                            break
                    if similarity > 0:
                        break
            
            if similarity > 0:
                results.append((concept1.id, concept2.id, similarity))
                
        except Exception:
            continue
    
    return results

class KnowledgeGraph:
    
    def __init__(self, graph_dir: Path = None, load_from_files: bool = True):
        
        self.graph_dir = graph_dir
        
        if self.graph_dir:
            self.graph_dir.mkdir(parents=True, exist_ok=True)
            
            self.nodes_file = self.graph_dir / "nodes.json"
            self.relations_file = self.graph_dir / "relations.json"
            self.term_mapping_file = self.graph_dir / "term_mapping.json"
            self.processing_records_file = self.graph_dir / "processing_records.json"
            self.file_node_mapping_file = self.graph_dir / "file_node_mapping.json"
        
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.relations: List[KnowledgeRelation] = []
        
        self.term_mapping: Dict[str, str] = {}
        
        self._relation_index = {}
        
        self.processing_records: Dict[str, FileProcessingRecord] = {}
        self.file_node_mapping: Dict[str, List[str]] = {}
        
        if load_from_files and self.graph_dir:
            self.load()
    
    def calculate_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""
    
    def load(self) -> None:
        try:
            if self.nodes_file.exists():
                with open(self.nodes_file, 'r', encoding='utf-8') as f:
                    nodes_data = json.load(f)
                    for node_data in nodes_data:
                        node = KnowledgeNode(
                            id=node_data['id'],
                            type=node_data['type'],
                            name=node_data['name'],
                            properties=node_data.get('properties', {}),
                            aliases=node_data.get('aliases', []),
                            name_ko=node_data.get('name_ko')
                        )
                        self.nodes[node.id] = node
            
            if self.relations_file.exists():
                with open(self.relations_file, 'r', encoding='utf-8') as f:
                    relations_data = json.load(f)
                    for rel_data in relations_data:
                        relation = KnowledgeRelation(
                            source=rel_data['source'],
                            target=rel_data['target'],
                            type=rel_data['type'],
                            weight=rel_data.get('weight', 1.0),
                            properties=rel_data.get('properties', {})
                        )
                        self.relations.append(relation)
                        
                        relation_key = f"{relation.source}:{relation.target}:{relation.type}"
                        self._relation_index[relation_key] = True
            
            if self.term_mapping_file.exists():
                with open(self.term_mapping_file, 'r', encoding='utf-8') as f:
                    self.term_mapping = json.load(f)
            
            if self.processing_records_file.exists():
                with open(self.processing_records_file, 'r', encoding='utf-8') as f:
                    records_data = json.load(f)
                    for file_path, record_data in records_data.items():
                        self.processing_records[file_path] = FileProcessingRecord(
                            file_path=record_data['file_path'],
                            file_hash=record_data['file_hash'],
                            processed_time=record_data['processed_time'],
                            doc_id=record_data['doc_id'],
                            nodes_created=record_data.get('nodes_created', []),
                            relations_created=record_data.get('relations_created', [])
                        )
            
            if self.file_node_mapping_file.exists():
                with open(self.file_node_mapping_file, 'r', encoding='utf-8') as f:
                    self.file_node_mapping = json.load(f)
            
            print(f"그래프 로드 완료: 노드 {len(self.nodes)}개, 관계 {len(self.relations)}개, "
                  f"용어 매핑 {len(self.term_mapping)}개, 처리 기록 {len(self.processing_records)}개")
        
        except Exception as e:
            print(f"그래프 로드 중 오류 발생: {str(e)}")
            self.nodes = {}
            self.relations = []
            self.term_mapping = {}
            self._relation_index = {}
            self.processing_records = {}
            self.file_node_mapping = {}
    
    def load_from_db(self, knowledge_data):
        try:
            nodes_data = knowledge_data.get('nodes', [])
            for node_data in nodes_data:
                node = KnowledgeNode(
                    id=node_data['id'],
                    type=node_data['type'],
                    name=node_data['name'],
                    properties=node_data.get('properties', {}),
                    aliases=node_data.get('aliases', []),
                    name_ko=node_data.get('name_ko')
                )
                self.nodes[node.id] = node

            relations_data = knowledge_data.get('relations', [])
            for rel_data in relations_data:
                relation = KnowledgeRelation(
                    source=rel_data['source'],
                    target=rel_data['target'],
                    type=rel_data['type'],
                    weight=rel_data.get('weight', 1.0),
                    properties=rel_data.get('properties', {})
                )
                self.relations.append(relation)

                relation_key = f"{relation.source}:{relation.target}:{relation.type}"
                self._relation_index[relation_key] = True

            self.term_mapping = knowledge_data.get('terms', {})

            records_data = knowledge_data.get('records', {})
            for file_path, record_data in records_data.items():
                self.processing_records[file_path] = FileProcessingRecord(
                    file_path=record_data['file_path'],
                    file_hash=record_data['file_hash'],
                    processed_time=record_data['processed_time'],
                    doc_id=record_data['doc_id'],
                    nodes_created=record_data.get('nodes_created', []),
                    relations_created=record_data.get('relations_created', [])
                )

            self.file_node_mapping = {}

            print(f"그래프 로드 완료: 노드 {len(self.nodes)}개, 관계 {len(self.relations)}개, "
                f"용어 매핑 {len(self.term_mapping)}개, 처리 기록 {len(self.processing_records)}개")

        except Exception as e:
            print(f"그래프 로드 중 오류 발생: {str(e)}")
            self.nodes = {}
            self.relations = []
            self.term_mapping = {}
            self._relation_index = {}
            self.processing_records = {}
            self.file_node_mapping = {}
    
    def save(self, sub_id, save_db = False) -> None:
        
        if save_db:
            try:
                from aibot_db_manager import AibotDBManager
                from aibot_db_command import SQL_QUERIES
                from config_utils import ConfigManager
                
                config_manager = ConfigManager()
                self.db_manager = AibotDBManager(
                    config=config_manager,
                    query_properties=SQL_QUERIES
                )
                print("📊 DB 모드 활성화: MariaDB 사용")
            except Exception as e:
                print(f"⚠️ DB 초기화 실패: {e}")
                print("파일 모드로 폴백합니다.")
                save_db = False
                self.db_manager = None
        
        try:
            print(f"노드 데이터 저장 중... ({len(self.nodes)}개)")
            nodes_data = []
            
            for node in tqdm(self.nodes.values(), desc="노드 저장", unit="node"):
                node_data = {
                    'id': node.id,
                    'type': node.type,
                    'name': node.name
                }
                
                if node.properties:
                    node_data['properties'] = node.properties
                
                if node.aliases:
                    node_data['aliases'] = node.aliases
                
                if node.name_ko:
                    node_data['name_ko'] = node.name_ko
                
                nodes_data.append(node_data)
            
            if not save_db:
                with open(self.nodes_file, 'w', encoding='utf-8') as f:
                    json.dump(nodes_data, f, ensure_ascii=False, indent=2)
            
            print(f"관계 데이터 저장 중... ({len(self.relations)}개)")
            relations_data = []
            
            for relation in tqdm(self.relations, desc="관계 저장", unit="relation"):
                rel_data = {
                    'source': relation.source,
                    'target': relation.target,
                    'type': relation.type,
                    'weight': relation.weight
                }
                
                if relation.properties:
                    rel_data['properties'] = relation.properties
                
                relations_data.append(rel_data)
            
            if not save_db:
                with open(self.relations_file, 'w', encoding='utf-8') as f:
                    json.dump(relations_data, f, ensure_ascii=False, indent=2)
            
            if not save_db:
                print(f"용어 매핑 저장 중... ({len(self.term_mapping)}개)")
                with open(self.term_mapping_file, 'w', encoding='utf-8') as f:
                    json.dump(self.term_mapping, f, ensure_ascii=False, indent=2)
            
            print(f"처리 기록 저장 중... ({len(self.processing_records)}개)")
            records_data = {}
            for file_path, record in self.processing_records.items():
                records_data[file_path] = {
                    'file_path': record.file_path,
                    'file_hash': record.file_hash,
                    'processed_time': record.processed_time,
                    'doc_id': record.doc_id,
                    'nodes_created': record.nodes_created,
                    'relations_created': record.relations_created
                }
            
            if not save_db:
                with open(self.processing_records_file, 'w', encoding='utf-8') as f:
                    json.dump(records_data, f, ensure_ascii=False, indent=2)
            
            with open(self.file_node_mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.file_node_mapping, f, ensure_ascii=False, indent=2)
                
            if save_db:
                self.db_manager.save_knowledge_graph(sub_id, "relations_data", json.dumps(relations_data, ensure_ascii=False, separators=(",", ":")))
                self.db_manager.save_knowledge_graph(sub_id, "records_data", json.dumps(records_data, ensure_ascii=False, separators=(",", ":")))
                self.db_manager.save_knowledge_graph(sub_id, "term_mapping", json.dumps(self.term_mapping, ensure_ascii=False, separators=(",", ":")))
                self.db_manager.save_knowledge_graph(sub_id, "nodes_data", json.dumps(nodes_data, ensure_ascii=False, separators=(",", ":")))
            
            print(f"그래프 저장 완료: 노드 {len(self.nodes)}개, 관계 {len(self.relations)}개, "
                  f"용어 매핑 {len(self.term_mapping)}개, 처리 기록 {len(self.processing_records)}개")
        
        except Exception as e:
            print(f"그래프 저장 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()

    def is_file_processed(self, file_path: Path) -> bool:
        file_path_str = str(file_path)
        
        if file_path_str not in self.processing_records:
            return False
        
        current_hash = self.calculate_file_hash(file_path)
        recorded_hash = self.processing_records[file_path_str].file_hash
        
        return current_hash == recorded_hash
    
    def remove_file_data(self, file_path: Path) -> None:
        file_path_str = str(file_path)
        
        if file_path_str not in self.processing_records:
            return
        
        record = self.processing_records[file_path_str]
        
        print(f"🗑️ 파일 관련 데이터 제거: {file_path.name}")
        
        nodes_removed = 0
        for node_id in record.nodes_created:
            if node_id in self.nodes:
                del self.nodes[node_id]
                nodes_removed += 1
        
        relations_to_remove = []
        relations_removed = 0
        
        for i, relation in enumerate(self.relations):
            if (relation.source in record.nodes_created or 
                relation.target in record.nodes_created):
                relations_to_remove.append(i)
                
                relation_key = f"{relation.source}:{relation.target}:{relation.type}"
                if relation_key in self._relation_index:
                    del self._relation_index[relation_key]
                
                relations_removed += 1
        
        for i in reversed(relations_to_remove):
            del self.relations[i]
        
        del self.processing_records[file_path_str]
        
        if file_path_str in self.file_node_mapping:
            del self.file_node_mapping[file_path_str]
        
        print(f"   제거 완료: 노드 {nodes_removed}개, 관계 {relations_removed}개")
    
    def remove_db_data(self, file_path_str: str) -> None:
        
        if file_path_str not in self.processing_records:
            return
        
        record = self.processing_records[file_path_str]
        
        print(f"🗑️ 파일 관련 데이터 제거: {file_path_str}")
        
        nodes_removed = 0
        for node_id in record.nodes_created:
            if node_id in self.nodes:
                del self.nodes[node_id]
                nodes_removed += 1
        
        relations_to_remove = []
        relations_removed = 0
        
        for i, relation in enumerate(self.relations):
            if (relation.source in record.nodes_created or 
                relation.target in record.nodes_created):
                relations_to_remove.append(i)
                
                relation_key = f"{relation.source}:{relation.target}:{relation.type}"
                if relation_key in self._relation_index:
                    del self._relation_index[relation_key]
                
                relations_removed += 1
        
        for i in reversed(relations_to_remove):
            del self.relations[i]
        
        del self.processing_records[file_path_str]
        
        if file_path_str in self.file_node_mapping:
            del self.file_node_mapping[file_path_str]
        
        print(f"   제거 완료: 노드 {nodes_removed}개, 관계 {relations_removed}개")
    
    def record_file_processing(self, file_path: Path, doc_id: str, 
                              nodes_created: List[str], relations_created: List[str]) -> None:
        file_path_str = str(file_path)
        
        record = FileProcessingRecord(
            file_path=file_path_str,
            file_hash=self.calculate_file_hash(file_path),
            processed_time=time.time(),
            doc_id=doc_id,
            nodes_created=nodes_created,
            relations_created=relations_created
        )
        
        self.processing_records[file_path_str] = record
        self.file_node_mapping[file_path_str] = nodes_created

    def add_node(self, node: KnowledgeNode) -> None:
        if node.id in self.nodes:
            existing_node = self.nodes[node.id]
            
            if node.name_ko and not existing_node.name_ko:
                existing_node.name_ko = node.name_ko
            
            for alias in node.aliases:
                if alias not in existing_node.aliases:
                    existing_node.aliases.append(alias)
            
            if node.properties:
                for key, value in node.properties.items():
                    existing_node.properties[key] = value
            
            return
        
        self.nodes[node.id] = node
        
        if node.name_ko:
            self.term_mapping[node.name.lower()] = node.name_ko
            self.term_mapping[node.name_ko.lower()] = node.name

    def add_relation(self, relation: KnowledgeRelation) -> None:
        if relation.source not in self.nodes or relation.target not in self.nodes:
            return
        
        relation_key = f"{relation.source}:{relation.target}:{relation.type}"
        
        if relation_key in self._relation_index:
            return
        
        self.relations.append(relation)
        self._relation_index[relation_key] = True
    
    def find_nodes_by_name(self, name: str, include_aliases: bool = True) -> List[KnowledgeNode]:
        name_lower = name.lower()
        results = []
        
        mapped_name = self.term_mapping.get(name_lower)
        
        name_no_space = name_lower.replace(" ", "")
        mapped_no_space = mapped_name.replace(" ", "") if mapped_name else None
        
        exact_matches = []
        for node in self.nodes.values():
            if name_lower == node.name.lower() or name_no_space == node.name.lower().replace(" ", ""):
                exact_matches.append(node)
                continue
            
            if node.name_ko and (name_lower == node.name_ko.lower() or 
                               name_no_space == node.name_ko.lower().replace(" ", "")):
                exact_matches.append(node)
                continue
            
            if mapped_name and (mapped_name.lower() == node.name.lower() or 
                              mapped_name.lower() == node.name_ko.lower() if node.name_ko else False):
                exact_matches.append(node)
                continue
            
            if include_aliases and node.aliases:
                for alias in node.aliases:
                    if name_lower == alias.lower() or name_no_space == alias.lower().replace(" ", ""):
                        exact_matches.append(node)
                        break
        
        if exact_matches:
            return exact_matches
        
        for node in self.nodes.values():
            if name_lower in node.name.lower() or name_no_space in node.name.lower().replace(" ", ""):
                results.append(node)
                continue
            
            if node.name_ko and (name_lower in node.name_ko.lower() or 
                               name_no_space in node.name_ko.lower().replace(" ", "")):
                results.append(node)
                continue
            
            if mapped_name and ((mapped_name.lower() in node.name.lower()) or 
                              (node.name_ko and mapped_name.lower() in node.name_ko.lower())):
                results.append(node)
                continue
            
            if include_aliases and node.aliases:
                for alias in node.aliases:
                    if name_lower in alias.lower() or name_no_space in alias.lower().replace(" ", ""):
                        results.append(node)
                        break
        
        return results
    
    def add_term_mapping(self, term1: str, term2: str) -> None:
        if not term1 or not term2:
            return
            
        term1 = term1.lower()
        term2 = term2.lower()
        
        self.term_mapping[term1] = term2
        self.term_mapping[term2] = term1


class KnowledgeGraphGenerator:
    
    def __init__(self, docs_dir: Path, graph_dir: Path):
        self.docs_dir = docs_dir
        self.graph_dir = graph_dir
        self.knowledge_graph = KnowledgeGraph(graph_dir)
        
        self.folder_groups = self._detect_folder_groups()
        
        self.term_mappings_file = self.graph_dir / "term_mappings.json"
        self.kr_en_mappings = self._load_term_mappings()
        
        if self.kr_en_mappings:
            self._apply_term_mappings()
    
    def analyze_file_changes(self, files: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        current_files = set(str(f) for f in files)
        processed_files = set(self.knowledge_graph.processing_records.keys())
        
        new_files = []
        for file_path in files:
            file_path_str = str(file_path)
            if file_path_str not in self.knowledge_graph.processing_records:
                new_files.append(file_path)
        
        modified_files = []
        for file_path in files:
            file_path_str = str(file_path)
            if (file_path_str in self.knowledge_graph.processing_records and 
                not self.knowledge_graph.is_file_processed(file_path)):
                modified_files.append(file_path)
        
        deleted_files = []
        for file_path_str in processed_files:
            if file_path_str not in current_files:
                if not Path(file_path_str).exists():
                    deleted_files.append(Path(file_path_str))
        
        return new_files, modified_files, deleted_files
    
    def process_update(self, files: Dict[str, str], sub_id, save_db, user_prompt=None) -> None:
        print("🔄 지식 그래프 증분 업데이트 분석 중...")
        
        print(f"📊 지식 그래프 변경사항:")
        file_count = len(files)
        print(f"   🆕 작업할 파일: {file_count}개")
        
        files_to_process = files
        
        if files_to_process:
            print(f"🚀 {len(files_to_process)}개 파일 처리 중...")
            processed_count = self.process_prompts(files_to_process)
            print(f"✅ 파일 처리 완료: {processed_count}/{file_count}개 성공")
            
            if processed_count > 0:
                paths = [Path(k) for k in files_to_process.keys()]
                self.establish_concept_relations_incremental(paths)
        
        self.knowledge_graph.save(sub_id, save_db)
        print("💾 지식 그래프 증분 업데이트 완료!")
    
    def process_incremental_update(self, files: List[Path], sub_id, save_db, user_prompt=None) -> None:
        print("🔄 지식 그래프 증분 업데이트 분석 중...")
        
        new_files, modified_files, deleted_files = self.analyze_file_changes(files)
        
        print(f"📊 지식 그래프 변경사항:")
        print(f"   🆕 새로운 파일: {len(new_files)}개")
        print(f"   ✏️ 수정된 파일: {len(modified_files)}개")
        print(f"   🗑️ 삭제된 파일: {len(deleted_files)}개")
        
        if not new_files and not modified_files and not deleted_files:
            print("✅ 지식 그래프 변경사항이 없습니다.")
            return
        
        for file_path in deleted_files:
            print(f"🗑️ 삭제된 파일 처리: {file_path.name}")
            self.knowledge_graph.remove_file_data(file_path)
        
        for file_path in modified_files:
            print(f"🔄 수정된 파일 재처리: {file_path.name}")
            self.knowledge_graph.remove_file_data(file_path)
        
        files_to_process = new_files + modified_files
        
        if files_to_process:
            print(f"🚀 {len(files_to_process)}개 파일 처리 중...")
            processed_count = self.process_files(files_to_process)
            print(f"✅ 파일 처리 완료: {processed_count}/{len(files_to_process)}개 성공")
            
            if processed_count > 0:
                self.establish_concept_relations_incremental(files_to_process)
        
        self.knowledge_graph.save(sub_id, save_db)
        print("💾 지식 그래프 증분 업데이트 완료!")
    
    def establish_concept_relations_incremental(self, updated_files: List[Path]) -> None:
        print("🔗 영향받은 개념들의 관계 재설정 중...")
        
        affected_node_ids = set()
        for file_path in updated_files:
            file_path_str = str(file_path)
            if file_path_str in self.knowledge_graph.file_node_mapping:
                affected_node_ids.update(self.knowledge_graph.file_node_mapping[file_path_str])
        
        if not affected_node_ids:
            print("   영향받은 노드가 없습니다.")
            return
        
        print(f"   영향받은 노드: {len(affected_node_ids)}개")
        
        relations_to_remove = []
        for i, relation in enumerate(self.knowledge_graph.relations):
            if (relation.source in affected_node_ids or 
                relation.target in affected_node_ids):
                relations_to_remove.append(i)
                
                relation_key = f"{relation.source}:{relation.target}:{relation.type}"
                if relation_key in self.knowledge_graph._relation_index:
                    del self.knowledge_graph._relation_index[relation_key]
        
        for i in reversed(relations_to_remove):
            del self.knowledge_graph.relations[i]
        
        print(f"   제거된 기존 관계: {len(relations_to_remove)}개")
        
        affected_concepts = [node_id for node_id in affected_node_ids 
                           if node_id in self.knowledge_graph.nodes and 
                           self.knowledge_graph.nodes[node_id].type == "concept"]
        
        if len(affected_concepts) < len(self.knowledge_graph.nodes) * 0.1:
            print(f"   부분 관계 업데이트: {len(affected_concepts)}개 개념")
            self._establish_relations_for_concepts(affected_concepts)
        else:
            print(f"   전체 관계 재설정: {len(self.knowledge_graph.nodes)}개 노드")
            self.establish_concept_relations()
    
    def _establish_relations_for_concepts(self, concept_ids: List[str]) -> None:
        if not concept_ids:
            return
        
        created_relations = 0
        all_concept_nodes = [node for node in self.knowledge_graph.nodes.values() 
                           if node.type == "concept"]
        
        for concept_id in tqdm(concept_ids, desc="개념 관계 재설정"):
            if concept_id not in self.knowledge_graph.nodes:
                continue
                
            concept = self.knowledge_graph.nodes[concept_id]
            
            for other_concept in all_concept_nodes:
                if other_concept.id == concept_id:
                    continue
                
                key1 = f"{concept_id}:{other_concept.id}:related"
                key2 = f"{other_concept.id}:{concept_id}:related"
                
                if key1 not in self.knowledge_graph._relation_index and key2 not in self.knowledge_graph._relation_index:
                    similarity = self._calculate_name_similarity(concept, other_concept)
                    
                    if similarity > 0.3:
                        relation = KnowledgeRelation(
                            source=concept_id,
                            target=other_concept.id,
                            type="related",
                            weight=similarity
                        )
                        
                        self.knowledge_graph.add_relation(relation)
                        created_relations += 1
        
        print(f"   생성된 새 관계: {created_relations}개")
    
    def _detect_folder_groups(self) -> List[str]:
        folder_groups = []
        
        yaml_dir = None
        
        for path in self.docs_dir.glob("**/yaml"):
            if path.is_dir():
                yaml_dir = path
                break
        
        if not yaml_dir and self.docs_dir.name.lower() == "yaml":
            yaml_dir = self.docs_dir
        
        if not yaml_dir:
            yaml_dir = self.docs_dir
        
        for folder in yaml_dir.iterdir():
            if folder.is_dir():
                folder_groups.append(folder.name)
        
        print(f"감지된 폴더 그룹: {folder_groups}")
        return folder_groups
    
    def _load_term_mappings(self) -> Dict[str, str]:
        if self.term_mappings_file.exists():
            try:
                with open(self.term_mappings_file, 'r', encoding='utf-8') as f:
                    mappings = json.load(f)
                print(f"용어 매핑 파일 로드: {self.term_mappings_file} ({len(mappings)}개 용어)")
                return mappings
            except Exception as e:
                print(f"용어 매핑 파일 로드 오류: {str(e)}")
        
        default_mappings = {
            "앱": "app",
            "어플": "app",
            "어플리케이션": "application",
            "애플리케이션": "application",
            "블루": "blue",
            "불루": "blue",
            "맥스": "max",
            "멕스": "max",
            "패스워드": "password",
            "비밀번호": "password",
            "아이피": "ip",
            "주소": "address",
            "명령어": "command",
            "설치": "install",
            "가이드": "guide",
            "유저": "user",
            "사용자": "user",
            "보고서": "report",
            "체크": "check",
            "확인": "check",
            "로그": "log",
            "기록": "log",
            "배치": "batch",
            "일괄": "batch",
            "블랙리스트": "blacklist",
            "차단목록": "blacklist",
            "머신러닝": "machine learning",
            "머신 러닝": "machine learning",
            "인공지능": "artificial intelligence",
            "딥러닝": "deep learning",
            "딥 러닝": "deep learning",
            "자연어처리": "natural language processing",
            "자연어 처리": "natural language processing"
        }
        
        print(f"기본 매핑 사용: {len(default_mappings)}개 용어")
        return default_mappings
    
    def _apply_term_mappings(self) -> None:
        count = 0
        for term1, term2 in self.kr_en_mappings.items():
            self.knowledge_graph.add_term_mapping(term1, term2)
            count += 1
        
        print(f"용어 매핑 {count}개 적용 완료")
    
    def normalize_term(self, term: str) -> str:
        if not term:
            return ""
            
        normalized = term.lower()
        
        normalized_no_space = normalized.replace(" ", "")
        
        if normalized in self.kr_en_mappings:
            return normalized
        
        if normalized_no_space in self.kr_en_mappings:
            return normalized_no_space
        
        return normalized
    
    def _normalize_id(self, text: str) -> str:
        normalized = re.sub(r'[^a-zA-Z0-9가-힣]', '_', text.lower())
        normalized = re.sub(r'_+', '_', normalized)
        if len(normalized) > 50:
            normalized = normalized[:50]
        return normalized
    
    def parse_filename(self, filename: str) -> Dict[str, Any]:
        parts = filename.split('-')
        
        result = {
            "full_name": filename,
            "header": parts[0].strip() if parts else "",
            "components": [p.strip() for p in parts[1:]] if len(parts) > 1 else [],
        }
        
        return result
    
    def detect_language(self, text: str) -> str:
        ko_chars = len(re.findall(r'[가-힣]', text))
        
        en_chars = len(re.findall(r'[A-Za-z]', text))
        
        if ko_chars > 0 and en_chars > 0:
            return 'mixed'
        elif ko_chars > en_chars:
            return 'ko'
        else:
            return 'en'
    
    def get_folder_group(self, file_path: Path) -> Optional[str]:
        try:
            rel_path = file_path.relative_to(self.docs_dir)
            path_parts = rel_path.parts
            
            for part in path_parts:
                if part in self.folder_groups:
                    return part
        except Exception:
            pass
            
        return None
    
    def process_file(self, file_path: Path) -> Optional[str]:
        if self.knowledge_graph.is_file_processed(file_path):
            return None
        
        try:
            relative_path = file_path.relative_to(self.docs_dir)
            doc_id = f"doc_{self._normalize_id(str(relative_path))}"
            
            self.knowledge_graph.remove_file_data(file_path)
            
            file_stem = file_path.stem
            parsed_filename = self.parse_filename(file_stem)
            
            folder_group = self.get_folder_group(file_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='cp949') as f:
                    content = f.read()
            
            try:
                data = yaml.safe_load(content)
                
                if not isinstance(data, dict):
                    return None
                
                nodes_created = []
                relations_created = []
                
                doc_name = data.get('name', file_stem.replace('-', ' ').title())
                
                aliases = data.get('aliases', [])
                if isinstance(aliases, str):
                    aliases = [aliases]
                
                doc_properties = {
                    "path": str(relative_path),
                    "filename": file_path.name,
                    "folder_group": folder_group,
                    "header": parsed_filename["header"],
                    "components": parsed_filename["components"]
                }
                
                doc_node = KnowledgeNode(
                    id=doc_id,
                    type="document",
                    name=doc_name,
                    properties=doc_properties,
                    aliases=aliases
                )
                
                lang = self.detect_language(doc_name)
                if lang == 'ko':
                    doc_node.name_ko = doc_name
                    
                    for kr_term, en_term in self.kr_en_mappings.items():
                        if kr_term in doc_name.lower():
                            en_name = doc_name.lower().replace(kr_term, en_term)
                            if en_name not in doc_node.aliases:
                                doc_node.aliases.append(en_name)
                            
                            doc_node.name = en_name.title()
                            break
                
                if aliases and not doc_node.name_ko:
                    for alias in aliases:
                        if self.detect_language(alias) == 'ko':
                            doc_node.name_ko = alias
                            break
                
                self.knowledge_graph.add_node(doc_node)
                nodes_created.append(doc_id)
                
                if 'question' in data and data['question']:
                    question_id = f"question_{self._normalize_id(file_stem)}"
                    question_text = data['question']
                    
                    question_node = KnowledgeNode(
                        id=question_id,
                        type="question",
                        name=question_text[:200] + ("..." if len(question_text) > 200 else ""),
                        properties={"source_doc": doc_id}
                    )
                    
                    self.knowledge_graph.add_node(question_node)
                    nodes_created.append(question_id)
                    
                    doc_question_relation = KnowledgeRelation(
                        source=doc_id,
                        target=question_id,
                        type="has_question",
                        weight=1.0
                    )
                    self.knowledge_graph.add_relation(doc_question_relation)
                    relations_created.append(f"{doc_id}:{question_id}:has_question")
                    
                    concepts = self._extract_concepts_from_text(question_text)
                    for concept_text in concepts:
                        if len(concept_text) < 3:
                            continue
                            
                        concept_id = f"concept_{self._normalize_id(concept_text)}"
                        
                        concept_node = KnowledgeNode(
                            id=concept_id,
                            type="concept",
                            name=concept_text.title(),
                            properties={"source_question": question_id}
                        )
                        
                        concept_lang = self.detect_language(concept_text)
                        if concept_lang == 'ko':
                            concept_node.name_ko = concept_text
                            
                            for kr_term, en_term in self.kr_en_mappings.items():
                                if kr_term in concept_text.lower():
                                    en_name = concept_text.lower().replace(kr_term, en_term)
                                    
                                    concept_node.name = en_name.title()
                                    
                                    concept_node.name_ko = concept_text
                                    break
                        
                        self.knowledge_graph.add_node(concept_node)
                        nodes_created.append(concept_id)
                        
                        question_concept_relation = KnowledgeRelation(
                            source=question_id,
                            target=concept_id,
                            type="mentions_concept",
                            weight=1.0
                        )
                        self.knowledge_graph.add_relation(question_concept_relation)
                        relations_created.append(f"{question_id}:{concept_id}:mentions_concept")
                        
                        doc_concept_relation = KnowledgeRelation(
                            source=doc_id,
                            target=concept_id,
                            type="contains_concept",
                            weight=1.0
                        )
                        self.knowledge_graph.add_relation(doc_concept_relation)
                        relations_created.append(f"{doc_id}:{concept_id}:contains_concept")
                
                header = parsed_filename["header"]
                if header:
                    header_id = f"header_{self._normalize_id(header)}"
                    
                    header_node = KnowledgeNode(
                        id=header_id,
                        type="header",
                        name=header.title(),
                        properties={"source": "filename"}
                    )
                    
                    header_lang = self.detect_language(header)
                    if header_lang == 'ko':
                        header_node.name_ko = header
                        
                        for kr_term, en_term in self.kr_en_mappings.items():
                            if kr_term in header.lower():
                                en_name = header.lower().replace(kr_term, en_term)
                                
                                header_node.name = en_name.title()
                                break
                    
                    self.knowledge_graph.add_node(header_node)
                    nodes_created.append(header_id)
                    
                    doc_header_relation = KnowledgeRelation(
                        source=doc_id,
                        target=header_id,
                        type="has_header",
                        weight=1.2
                    )
                    self.knowledge_graph.add_relation(doc_header_relation)
                    relations_created.append(f"{doc_id}:{header_id}:has_header")
                
                for idx, component in enumerate(parsed_filename["components"]):
                    component_id = f"component_{self._normalize_id(component)}"
                    
                    component_node = KnowledgeNode(
                        id=component_id,
                        type="component",
                        name=component.title(),
                        properties={"source": "filename", "position": idx}
                    )
                    
                    comp_lang = self.detect_language(component)
                    if comp_lang == 'ko':
                        component_node.name_ko = component
                        
                        for kr_term, en_term in self.kr_en_mappings.items():
                            if kr_term in component.lower():
                                en_name = component.lower().replace(kr_term, en_term)
                                component_node.name = en_name.title()
                                break
                    
                    self.knowledge_graph.add_node(component_node)
                    nodes_created.append(component_id)
                    
                    doc_component_relation = KnowledgeRelation(
                        source=doc_id,
                        target=component_id,
                        type="has_component",
                        weight=1.0 + (0.1 * idx)
                    )
                    self.knowledge_graph.add_relation(doc_component_relation)
                    relations_created.append(f"{doc_id}:{component_id}:has_component")
                    
                    if header:
                        header_component_relation = KnowledgeRelation(
                            source=header_id,
                            target=component_id,
                            type="contains_component",
                            weight=1.0
                        )
                        self.knowledge_graph.add_relation(header_component_relation)
                        relations_created.append(f"{header_id}:{component_id}:contains_component")
                
                self.knowledge_graph.record_file_processing(
                    file_path, doc_id, nodes_created, relations_created
                )
                
                return doc_id
                
            except Exception as e:
                print(f"YAML 파싱 오류 ({file_path}): {str(e)}")
                return None
                
        except Exception as e:
            print(f"파일 처리 오류 ({file_path}): {str(e)}")
            return None
        
    def process_file_for_user_prompt(self, path, content) -> Optional[str]:
        value = content
        file_path = Path(path)
        
        
        try:
            doc_id = f"doc_{self._normalize_id(str(file_path))}"
            
            self.knowledge_graph.remove_file_data(file_path)
            
            file_stem = file_path.stem
            parsed_filename = self.parse_filename(file_stem)
            
            folder_group = self.get_folder_group(file_path)
            
            try:
                data = yaml.safe_load(value)
                if not isinstance(data, dict):
                    print("not isinstance(data, dict)")
                    return None
                
                nodes_created = []
                relations_created = []
                
                doc_name = data.get('name', file_stem.replace('-', ' ').title())
                
                aliases = data.get('aliases', [])
                if isinstance(aliases, str):
                    aliases = [aliases]
                
                doc_properties = {
                    "path": str(file_path),
                    "filename": file_path.name,
                    "folder_group": folder_group,
                    "header": parsed_filename["header"],
                    "components": parsed_filename["components"]
                }
                
                doc_node = KnowledgeNode(
                    id=doc_id,
                    type="document",
                    name=doc_name,
                    properties=doc_properties,
                    aliases=aliases
                )
                
                lang = self.detect_language(doc_name)
                if lang == 'ko':
                    doc_node.name_ko = doc_name
                    
                    for kr_term, en_term in self.kr_en_mappings.items():
                        if kr_term in doc_name.lower():
                            en_name = doc_name.lower().replace(kr_term, en_term)
                            if en_name not in doc_node.aliases:
                                doc_node.aliases.append(en_name)
                            
                            doc_node.name = en_name.title()
                            break
                
                if aliases and not doc_node.name_ko:
                    for alias in aliases:
                        if self.detect_language(alias) == 'ko':
                            doc_node.name_ko = alias
                            break
                
                self.knowledge_graph.add_node(doc_node)
                nodes_created.append(doc_id)
                
                if 'question' in data and data['question']:
                    question_id = f"question_{self._normalize_id(file_stem)}"
                    question_text = data['question']
                    
                    question_node = KnowledgeNode(
                        id=question_id,
                        type="question",
                        name=question_text[:200] + ("..." if len(question_text) > 200 else ""),
                        properties={"source_doc": doc_id}
                    )
                    
                    self.knowledge_graph.add_node(question_node)
                    nodes_created.append(question_id)
                    
                    doc_question_relation = KnowledgeRelation(
                        source=doc_id,
                        target=question_id,
                        type="has_question",
                        weight=1.0
                    )
                    self.knowledge_graph.add_relation(doc_question_relation)
                    relations_created.append(f"{doc_id}:{question_id}:has_question")
                    
                    concepts = self._extract_concepts_from_text(question_text)
                    for concept_text in concepts:
                        if len(concept_text) < 3:
                            continue
                            
                        concept_id = f"concept_{self._normalize_id(concept_text)}"
                        
                        concept_node = KnowledgeNode(
                            id=concept_id,
                            type="concept",
                            name=concept_text.title(),
                            properties={"source_question": question_id}
                        )
                        
                        concept_lang = self.detect_language(concept_text)
                        if concept_lang == 'ko':
                            concept_node.name_ko = concept_text
                            
                            for kr_term, en_term in self.kr_en_mappings.items():
                                if kr_term in concept_text.lower():
                                    en_name = concept_text.lower().replace(kr_term, en_term)
                                    
                                    concept_node.name = en_name.title()
                                    
                                    concept_node.name_ko = concept_text
                                    break
                        
                        self.knowledge_graph.add_node(concept_node)
                        nodes_created.append(concept_id)
                        
                        question_concept_relation = KnowledgeRelation(
                            source=question_id,
                            target=concept_id,
                            type="mentions_concept",
                            weight=1.0
                        )
                        self.knowledge_graph.add_relation(question_concept_relation)
                        relations_created.append(f"{question_id}:{concept_id}:mentions_concept")
                        
                        doc_concept_relation = KnowledgeRelation(
                            source=doc_id,
                            target=concept_id,
                            type="contains_concept",
                            weight=1.0
                        )
                        self.knowledge_graph.add_relation(doc_concept_relation)
                        relations_created.append(f"{doc_id}:{concept_id}:contains_concept")
                
                header = parsed_filename["header"]
                if header:
                    header_id = f"header_{self._normalize_id(header)}"
                    
                    header_node = KnowledgeNode(
                        id=header_id,
                        type="header",
                        name=header.title(),
                        properties={"source": "filename"}
                    )
                    
                    header_lang = self.detect_language(header)
                    if header_lang == 'ko':
                        header_node.name_ko = header
                        
                        for kr_term, en_term in self.kr_en_mappings.items():
                            if kr_term in header.lower():
                                en_name = header.lower().replace(kr_term, en_term)
                                
                                header_node.name = en_name.title()
                                break
                    
                    self.knowledge_graph.add_node(header_node)
                    nodes_created.append(header_id)
                    
                    doc_header_relation = KnowledgeRelation(
                        source=doc_id,
                        target=header_id,
                        type="has_header",
                        weight=1.2
                    )
                    self.knowledge_graph.add_relation(doc_header_relation)
                    relations_created.append(f"{doc_id}:{header_id}:has_header")
                
                for idx, component in enumerate(parsed_filename["components"]):
                    component_id = f"component_{self._normalize_id(component)}"
                    
                    component_node = KnowledgeNode(
                        id=component_id,
                        type="component",
                        name=component.title(),
                        properties={"source": "filename", "position": idx}
                    )
                    
                    comp_lang = self.detect_language(component)
                    if comp_lang == 'ko':
                        component_node.name_ko = component
                        
                        for kr_term, en_term in self.kr_en_mappings.items():
                            if kr_term in component.lower():
                                en_name = component.lower().replace(kr_term, en_term)
                                component_node.name = en_name.title()
                                break
                    
                    self.knowledge_graph.add_node(component_node)
                    nodes_created.append(component_id)
                    
                    doc_component_relation = KnowledgeRelation(
                        source=doc_id,
                        target=component_id,
                        type="has_component",
                        weight=1.0 + (0.1 * idx)
                    )
                    self.knowledge_graph.add_relation(doc_component_relation)
                    relations_created.append(f"{doc_id}:{component_id}:has_component")
                    
                    if header:
                        header_component_relation = KnowledgeRelation(
                            source=header_id,
                            target=component_id,
                            type="contains_component",
                            weight=1.0
                        )
                        self.knowledge_graph.add_relation(header_component_relation)
                        relations_created.append(f"{header_id}:{component_id}:contains_component")
                
                self.knowledge_graph.record_file_processing(
                    file_path, doc_id, nodes_created, relations_created
                )
                
                return doc_id
                
            except Exception as e:
                print(f"YAML 파싱 오류 ({file_path}): {str(e)}")
                return None
                
        except Exception as e:
            print(f"파일 처리 오류 ({file_path}): {str(e)}")
            return None
    
    def _extract_concepts_from_text(self, text: str) -> List[str]:
        concepts = []
        
        is_korean = bool(re.search(r'[가-힣]', text))
        
        korean_stop_words = set(['이', '그', '저', '것', '은', '는', '이', '가', '를', '에', '와', '과', '이다', '있다'])
        english_stop_words = set(['the', 'a', 'an', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about'])
        
        if is_korean:
            words = [word.strip() for word in text.split() if len(word) > 1]
            
            filtered_words = []
            for word in words:
                if word not in korean_stop_words:
                    word = re.sub(r'(이|가|을|를|에|의|로|으로|와|과|이나|나|은|는)$', '', word)
                    if len(word) > 1:
                        filtered_words.append(word)
            
            for i in range(len(filtered_words) - 1):
                phrase = filtered_words[i] + ' ' + filtered_words[i + 1]
                if len(phrase) > 3:
                    concepts.append(phrase)
            
            concepts.extend([w for w in filtered_words if len(w) > 1])
        else:
            words = [word.strip().lower() for word in text.split() if len(word) > 2]
            
            filtered_words = [word for word in words if word not in english_stop_words]
            
            noun_phrases = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+){0,2}', text)
            if noun_phrases:
                concepts.extend(noun_phrases)
            
            for i in range(len(filtered_words) - 1):
                phrase = filtered_words[i] + ' ' + filtered_words[i + 1]
                if len(phrase) > 4:
                    concepts.append(phrase)
            
            concepts.extend([w for w in filtered_words if len(w) > 2])
        
        normalized_concepts = []
        seen = set()
        
        for concept in concepts:
            norm_concept = self.normalize_term(concept)
            
            if norm_concept not in seen and len(norm_concept) > 2:
                seen.add(norm_concept)
                normalized_concepts.append(concept)
        
        return normalized_concepts[:10]
    
    def process_files(self, files: List[Path]) -> int:
        print(f"파일 처리 시작: {len(files)}개 파일")
        
        processed_count = 0
        for file_path in tqdm(files, desc="파일 처리 중"):
            try:
                doc_id = self.process_file(file_path)
                if doc_id:
                    processed_count += 1
            except Exception as e:
                print(f"파일 처리 오류 ({file_path.name}): {str(e)}")
        
        print(f"파일 처리 완료: {processed_count}/{len(files)}개 파일 성공")
        return processed_count
    
    def process_prompts(self, prompts: Dict[str, str]) -> int:
        total = len(prompts)
        print(f"파일 처리 시작: {total}개 파일")

        processed_count = 0
        failures: List[str] = []

        for path, content in tqdm(prompts.items(), total=total, desc="파일 처리 중", unit="file"):
            try:
                doc_id: Optional[str] = self.process_file_for_user_prompt(path, content)
                if doc_id:
                    processed_count += 1
                else:
                    failures.append(path)
            except Exception as e:
                failures.append(path)
                print(f"파일 처리 오류 ({path}): {e}")

        print(f"파일 처리 완료: {processed_count}/{total}개 파일 성공")
        if failures:
            print(f"실패 {len(failures)}개: {failures}")
        return processed_count
    
    def process_directory(self, directory: Path = None) -> int:
        if directory is None:
            directory = self.docs_dir
        
        print(f"디렉토리 처리: {directory}")
        
        yaml_files = list(directory.glob("**/*.yaml"))
        yaml_files.extend(list(directory.glob("**/*.yml")))
        
        if not yaml_files:
            print("처리할 YAML 파일이 없습니다.")
            return 0
        
        print(f"발견된 YAML 파일: {len(yaml_files)}개")
        
        self.process_incremental_update(yaml_files)
        
        return len(yaml_files)
    
    def establish_concept_relations(self) -> None:
        print("개념 간 관계 설정 중...")
        
        doc_concepts = {}
        concept_docs = {}
        
        relation_index = self.knowledge_graph._relation_index.copy()
        
        for relation in self.knowledge_graph.relations:
            if relation.type == "contains_concept":
                doc_id = relation.source
                concept_id = relation.target
                
                if doc_id not in doc_concepts:
                    doc_concepts[doc_id] = []
                doc_concepts[doc_id].append(concept_id)
                
                if concept_id not in concept_docs:
                    concept_docs[concept_id] = []
                concept_docs[concept_id].append(doc_id)
        
        created_relations = 0
        
        print(f"동일 문서 내 개념 관계 설정 중... (문서 수: {len(doc_concepts)})")
        
        total_concept_pairs = 0
        for concepts in doc_concepts.values():
            if len(concepts) >= 2:
                n = len(concepts)
                total_concept_pairs += (n * (n - 1)) // 2
        
        with tqdm(total=total_concept_pairs, desc="개념 관계 설정") as pbar:
            for doc_id, concepts in doc_concepts.items():
                if len(concepts) < 2:
                    continue
                
                for i in range(len(concepts)):
                    for j in range(i + 1, len(concepts)):
                        concept1_id = concepts[i]
                        concept2_id = concepts[j]
                        
                        key1 = f"{concept1_id}:{concept2_id}:related"
                        key2 = f"{concept2_id}:{concept1_id}:related"
                        
                        if key1 not in relation_index and key2 not in relation_index:
                            relation1 = KnowledgeRelation(
                                source=concept1_id,
                                target=concept2_id,
                                type="related",
                                weight=1.0
                            )
                            relation2 = KnowledgeRelation(
                                source=concept2_id,
                                target=concept1_id,
                                type="related",
                                weight=1.0
                            )
                            
                            self.knowledge_graph.add_relation(relation1)
                            self.knowledge_graph.add_relation(relation2)
                            
                            relation_index[key1] = True
                            relation_index[key2] = True
                            
                            created_relations += 2
                        
                        pbar.update(1)
        
        concept_nodes = sorted(
            [node for node in self.knowledge_graph.nodes.values() if node.type == "concept"],
            key=lambda x: x.id
        )
        
        print(f"🚀 병렬 이름 유사도 기반 개념 관계 설정 중... (개념 수: {len(concept_nodes)})")
        
        if len(concept_nodes) < 2:
            print("개념이 2개 미만이므로 유사도 관계 설정을 건너뜁니다.")
            return
        
        all_pairs = []
        for i in range(len(concept_nodes)):
            for j in range(i + 1, len(concept_nodes)):
                all_pairs.append((concept_nodes[i], concept_nodes[j]))
        
        print(f"   🔢 총 비교 쌍: {len(all_pairs):,}개")
        
        if not all_pairs:
            print("   ⚠️ 비교할 쌍이 없습니다.")
            return
        
        batch_size = 1000
        batches = []
        threshold = 0.7
        
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i + batch_size]
            batches.append((batch, threshold))
        
        workers = min(multiprocessing.cpu_count(), 24)
        print(f"   🚀 병렬 처리 시작 ({workers}개 워커, {len(batches)}개 배치)")
        
        all_results = []
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_similarity_batch, batch_args) for batch_args in batches]
            
            with tqdm(total=len(batches), desc="유사도 계산", ncols=80) as pbar:
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)
                        pbar.set_postfix({'유사쌍': len(all_results)})
                        pbar.update(1)
                    except Exception as e:
                        print(f"배치 처리 오류: {e}")
                        pbar.update(1)
        
        unique_results = {}
        for concept1_id, concept2_id, sim in all_results:
            key = tuple(sorted([concept1_id, concept2_id]))
            if key not in unique_results or unique_results[key] < sim:
                unique_results[key] = sim
        
        similar_relations = 0
        for (concept1_id, concept2_id), similarity in unique_results.items():
            try:
                key1 = f"{concept1_id}:{concept2_id}:similar"
                key2 = f"{concept2_id}:{concept1_id}:similar"
                
                if key1 not in relation_index and key2 not in relation_index:
                    relation1 = KnowledgeRelation(
                        source=concept1_id,
                        target=concept2_id,
                        type="similar",
                        weight=similarity
                    )
                    relation2 = KnowledgeRelation(
                        source=concept2_id,
                        target=concept1_id,
                        type="similar",
                        weight=similarity
                    )
                    
                    self.knowledge_graph.add_relation(relation1)
                    self.knowledge_graph.add_relation(relation2)
                    
                    relation_index[key1] = True
                    relation_index[key2] = True
                    
                    similar_relations += 2
                
            except Exception as e:
                print(f"관계 추가 오류: {e}")
                continue
        
        processing_time = time.time() - start_time
        print(f"   ✅ 완료: {processing_time:.1f}초, {similar_relations}개 유사도 관계 추가")
        print(f"   ⚡ 성능: {len(all_pairs)/processing_time:.0f} 비교/초")
        
        print(f"개념 관계 설정 완료: {created_relations}개 일반 관계, {similar_relations}개 유사도 관계 생성")

    def _calculate_name_similarity(self, node1: KnowledgeNode, node2: KnowledgeNode) -> float:
        name1 = self.normalize_term(node1.name)
        name2 = self.normalize_term(node2.name)
        
        if name1 == name2:
            return 1.0
        
        if node1.name_ko and node2.name_ko:
            ko_name1 = self.normalize_term(node1.name_ko)
            ko_name2 = self.normalize_term(node2.name_ko)
            
            if ko_name1 == ko_name2:
                return 1.0
        
        for alias1 in node1.aliases:
            norm_alias1 = self.normalize_term(alias1)
            
            if norm_alias1 == name2:
                return 0.9
            
            if node2.name_ko and norm_alias1 == self.normalize_term(node2.name_ko):
                return 0.9
            
            for alias2 in node2.aliases:
                if norm_alias1 == self.normalize_term(alias2):
                    return 0.8
        
        if name1 in name2 or name2 in name1:
            return 0.7
        
        try:
            import difflib
            ratio = difflib.SequenceMatcher(None, name1, name2).ratio()
            return ratio if ratio > 0.5 else 0.0
        except:
            set1 = set(name1)
            set2 = set(name2)
            
            if not set1 or not set2:
                return 0.0
                
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union if union > 0 else 0.0
    
    def create_knowledge_graph(self, sub_id) -> None:
        try:
            print("지식 그래프 생성 시작...")
            
            processed_count = self.process_directory()
            print(f"파일 처리 완료: {processed_count}개 파일 처리됨")
            
            if processed_count > 0:
                self.establish_concept_relations()
            
            self.knowledge_graph.save(sub_id)
            
            print("지식 그래프 생성 완료:")
            print(f"- 노드 수: {len(self.knowledge_graph.nodes)}개")
            print(f"- 관계 수: {len(self.knowledge_graph.relations)}개")
            print(f"- 용어 매핑 수: {len(self.knowledge_graph.term_mapping)}개")
            print(f"- 처리 기록 수: {len(self.knowledge_graph.processing_records)}개")
            
        except Exception as e:
            print(f"지식 그래프 생성 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()


def generate_kr_en_mappings(yaml_files: List[Path]) -> Dict[str, str]:
    print("한영 용어 매핑 생성 중...")
    
    default_mappings = {
        "앱": "app",
        "어플": "app",
        "어플리케이션": "application",
        "애플리케이션": "application",
        "블루": "blue",
        "불루": "blue",
        "맥스": "max",
        "멕스": "max",
        "패스워드": "password",
        "비밀번호": "password",
        "아이피": "ip",
        "주소": "address",
        "명령어": "command",
        "설치": "install",
        "가이드": "guide",
        "유저": "user",
        "사용자": "user",
        "보고서": "report",
        "체크": "check",
        "확인": "check",
        "로그": "log",
        "기록": "log",
        "배치": "batch",
        "일괄": "batch",
        "블랙리스트": "blacklist",
        "차단목록": "blacklist",
        "머신러닝": "machine learning",
        "머신 러닝": "machine learning",
        "인공지능": "artificial intelligence",
        "딥러닝": "deep learning",
        "딥 러닝": "deep learning",
        "자연어처리": "natural language processing",
        "자연어 처리": "natural language processing"
    }
    
    def is_korean(text):
        return bool(re.search(r'[가-힣]', text))
    
    def is_english(text):
        return bool(re.match(r'^[a-zA-Z\s]+$', text))
    
    name_aliases_pairs = []
    
    for file_path in tqdm(yaml_files, desc="파일 분석"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                continue
            
            if 'name' in data:
                name = data['name']
                aliases = data.get('aliases', [])
                
                if isinstance(aliases, str):
                    aliases = [aliases]
                
                if name and aliases:
                    name_aliases_pairs.append((name, aliases))
        except Exception:
            continue
    
    kr_en_candidates = {}
    
    for name, aliases in name_aliases_pairs:
        if is_korean(name):
            for alias in aliases:
                if is_english(alias):
                    kr_en_candidates[name.lower()] = alias.lower()
                    break
        
        elif is_english(name):
            for alias in aliases:
                if is_korean(alias):
                    kr_en_candidates[alias.lower()] = name.lower()
                    break
    
    for file_path in yaml_files:
        file_stem = file_path.stem
        parts = file_stem.split('-')
        
        for i in range(len(parts) - 1):
            if is_korean(parts[i]) and is_english(parts[i+1]):
                kr_en_candidates[parts[i].lower()] = parts[i+1].lower()
            elif is_english(parts[i]) and is_korean(parts[i+1]):
                kr_en_candidates[parts[i+1].lower()] = parts[i].lower()
    
    final_mappings = {**default_mappings, **kr_en_candidates}
    
    space_variations = {}
    for kr, en in final_mappings.items():
        if ' ' in kr:
            space_variations[kr.replace(' ', '')] = en
        
        if ' ' in en:
            space_variations[kr] = en.replace(' ', '')
    
    final_mappings.update(space_variations)
    
    print(f"한영 용어 매핑 생성 완료: {len(final_mappings)}개 매핑")
    return final_mappings

def save_kr_en_mappings(mappings: Dict[str, str], file_path: Path) -> None:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        
        print(f"한영 매핑 파일 저장 완료: {file_path}")
    except Exception as e:
        print(f"한영 매핑 파일 저장 오류: {str(e)}")

def create_knowledge_graph(docs_dir, graph_dir, files):
    try:
        graph_dir.mkdir(parents=True, exist_ok=True)
        
        generator = KnowledgeGraphGenerator(docs_dir, graph_dir)
        
        print(f"지식 그래프 증분 처리: {len(files)}개 파일")
        
        generator.process_incremental_update(files)
        
        print(f"지식 그래프 처리 완료:")
        print(f"- 노드 수: {len(generator.knowledge_graph.nodes)}개")
        print(f"- 관계 수: {len(generator.knowledge_graph.relations)}개")
        print(f"- 처리 기록 수: {len(generator.knowledge_graph.processing_records)}개")
        
        return generator.knowledge_graph
        
    except Exception as e:
        print(f"지식 그래프 생성 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    docs_directory = Path("/data/ai_qa/docs/aibot/yaml")
    graph_directory = Path("/data/ai_qa/qa_embeddings/knowledge_graph")
    
    if not docs_directory.exists():
        print(f"오류: 문서 디렉토리가 존재하지 않습니다: {docs_directory}")
        return
    
    graph_directory.mkdir(parents=True, exist_ok=True)
    
    yaml_files = list(docs_directory.glob("**/*.yaml"))
    yaml_files.extend(list(docs_directory.glob("**/*.yml")))
    
    if not yaml_files:
        print("오류: 처리할 YAML 파일이 없습니다.")
        return
    
    print(f"발견된 YAML 파일: {len(yaml_files)}개")
    
    mappings_file = graph_directory / "term_mappings.json"
    if not mappings_file.exists():
        print("🔄 한영 매핑 생성 중...")
        mappings = generate_kr_en_mappings(yaml_files)
        save_kr_en_mappings(mappings, mappings_file)
    else:
        print("✅ 기존 한영 매핑 사용")
    
    graph_generator = KnowledgeGraphGenerator(docs_directory, graph_directory)
    
    graph_generator.process_incremental_update(yaml_files)
    
    print("✅ 지식 그래프 증분 처리 완료!")


if __name__ == "__main__":
    main()
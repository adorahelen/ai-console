
SQL_QUERIES = {
    'CHECK_TABLES': """
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema = DATABASE() AND table_name = %s
    """,
    
    'COUNT_SQL': """
        SELECT COUNT(*) AS cnt
        FROM openai_prompts p
        LEFT JOIN ai_subscriptions s ON p.subscription_id = s.id
    """,
    
    'INSERT_PROMPT': """
        INSERT INTO openai_prompts
            (guid, subscription_id, name, description, type, prompt, token_count, vector, question_vector, source, tags)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            name = VALUES(name),
            description = VALUES(description),
            type = VALUES(type),
            prompt = VALUES(prompt),
            token_count = VALUES(token_count),
            vector = VALUES(vector),
            question_vector = VALUES(question_vector),
            source = VALUES(source),
            tags = VALUES(tags)
    """,
    
    'UPDATE_PROMPT': """
        UPDATE openai_prompts
        SET name=%s, description=%s, type=%s, rev=rev+1, prompt=%s
        WHERE guid=%s
    """,
    
    'UPDATE_PROMPT_VECTOR': """
        UPDATE openai_prompts
        SET name=%s, description=%s, type=%s, prompt=%s, token_count=%s, rev=rev+1, vector=%s, question_vector=%s, source=%s, tags=%s
        WHERE guid=%s
    """,
    
    'DELETE_PROMPT': """
        DELETE FROM openai_prompts WHERE guid IN (%s)
    """,
    
    'PROMPT_SQL': """
        SELECT p.id, p.guid, s.id as subscription_id, s.guid as subscription_guid,
            s.name as subscription_name, p.name, p.description, p.enabled, p.type,
            p.prompt, p.token_count, p.vector, p.question_vector, p.rev, p.source, p.created_at, p.updated_at
        FROM openai_prompts p
        LEFT JOIN ai_subscriptions s ON p.subscription_id = s.id
    """,

    'PROMPT_TINY_SQL': """
        SELECT p.id, p.guid, s.id as subscription_id, s.guid as subscription_guid,
            s.name as subscription_name, p.name, p.description, p.enabled, p.type,
            p.rev, p.created_at, p.updated_at
        FROM openai_prompts p
        LEFT JOIN ai_subscriptions s ON p.subscription_id = s.id
    """,
    
    'INSERT_FAISS_INDEX': """
        INSERT INTO ai_faiss_indices (subscription_id, index_name, index_data, metadata, artifact_metadata)
        VALUES (%s, %s, %s, %s, %s)
    """,
    
    'INSERT_FAISS_PARTS': """
        INSERT INTO ai_faiss_index_parts
        (subscription_id, index_name, part_no, chunk)
        VALUES (%s,%s,%s,%s)
    """,
    
    'UPDATE_FAISS_INDEX': """
        UPDATE ai_faiss_indices
        SET index_data = %s, metadata = %s, artifact_metadata = %s, rev=rev+1, updated_at = NOW()
        WHERE subscription_id = %s AND index_name = %s
    """,
    
    'DELETE_FAISS_INDEX': """
        DELETE FROM ai_faiss_indices WHERE subscription_id = %s
    """,
    
    'DELETE_FAISS_PARTS': """
        DELETE FROM ai_faiss_index_parts WHERE subscription_id=%s AND index_name=%s
    """,
    
    'GET_FAISS_INDEX_BY_NAME': """
        SELECT id, subscription_id, index_name, index_data, metadata, artifact_metadata,
               created_at, updated_at
        FROM ai_faiss_indices
        WHERE subscription_id = %s AND index_name = %s
    """,
    
    'GET_ALL_FAISS_INDEXES': """
        SELECT id, subscription_id, index_name, index_data, metadata, artifact_metadata, intent, created_at, updated_at
        FROM ai_faiss_indices
        WHERE subscription_id = %s
        ORDER BY updated_at DESC
    """,
    
    'GET_FAISS_PARTS': """
        SELECT part_no, chunk FROM ai_faiss_index_parts
        WHERE subscription_id=%s AND index_name=%s
        ORDER BY part_no ASC
    """,
    
    # ========== 지식 그래프 쿼리 ==========
    'INSERT_KNOWLEDGE_GRAPH': """
        INSERT INTO ai_knowledge_graph (subscription_id, data_name, metadata) 
        VALUES (
            %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            metadata = VALUES(metadata),
            rev=rev+1,
            updated_at = CURRENT_TIMESTAMP(6);
    """,
    
    'GET_KNOWLEDGE_GRAPH': """
        SELECT subscription_id, data_name, metadata
        FROM ai_knowledge_graph
    """,
    
    'INSERT_KG_CHUNK': """
        INSERT INTO ai_knowledge_graph (subscription_id, data_name, metadata, rev)
        VALUES (%s, %s, JSON_OBJECT('storage','parts','codec','gzip','orig_size',%s,'parts',%s,'sha256',%s), 1)
        ON DUPLICATE KEY UPDATE
        metadata = VALUES(metadata),
        rev = rev + 1
    """,
    
    'GET_KG_ID': """
        SELECT id FROM ai_knowledge_graph
        WHERE subscription_id <=> %s AND data_name = %s
        ORDER BY id DESC LIMIT 1
    """,
    
    'DEL_CHUNK_DATA': """
        DELETE FROM ai_knowledge_graph_parts WHERE kg_id=%s
    """,
    
    'INSERT_CHUNK_DATA': """
        INSERT INTO ai_knowledge_graph_parts (kg_id, part_no, codec, chunk, checksum)
        VALUES (%s, %s, 'gzip', %s, %s)
    """,
    
    'GET_METADATA_INFO': """
        SELECT id, metadata
            FROM ai_knowledge_graph
        WHERE subscription_id <=> %s AND data_name = %s
        LIMIT 1
    """,
    
    'GET_METADATA': """
        SELECT part_no, codec, chunk, checksum
            FROM ai_knowledge_graph_parts
        WHERE kg_id = %s
        ORDER BY part_no ASC
    """,
    
    'DELETE_KG_ID': """
        DELETE FROM ai_knowledge_graph WHERE subscription_id = %s
    """,
    
    'DELETE_OLD_DOCUMENTS': """
        DELETE FROM documents 
        WHERE updated_at < DATE_SUB(NOW(), INTERVAL %s DAY)
    """,
    
    'DELETE_ORPHAN_EMBEDDINGS': """
        DELETE e FROM embeddings e
        LEFT JOIN documents d ON e.doc_id = d.doc_id
        WHERE d.doc_id IS NULL
    """,
    
    'GET_DOCUMENT_COUNT': """
        SELECT COUNT(*) as count FROM documents
    """,
    
    'GET_EMBEDDING_COUNT': """
        SELECT COUNT(*) as count FROM embeddings
    """,
    
    'GET_EMBEDDING_COUNT_BY_DOCUMENT': """
        SELECT doc_id, COUNT(*) as embedding_count
        FROM embeddings
        GROUP BY doc_id
    """,
    
    'OPTIMIZE_TABLES': """
        OPTIMIZE TABLE documents, embeddings, faiss_indexes
    """,
    
    'SEARCH_DOCUMENTS_BY_DATE_RANGE': """
        SELECT doc_id, doc_name, 
               SUBSTRING(content, 1, 500) as content_preview,
               metadata, created_at, updated_at
        FROM documents
        WHERE created_at BETWEEN %s AND %s
        ORDER BY created_at DESC
        LIMIT %s
    """,
    
    'SEARCH_DOCUMENTS_BY_METADATA': """
        SELECT doc_id, doc_name, 
               SUBSTRING(content, 1, 500) as content_preview,
               metadata, created_at, updated_at
        FROM documents
        WHERE JSON_CONTAINS(metadata, %s)
        ORDER BY updated_at DESC
        LIMIT %s
    """,
    
    'GET_RECENT_DOCUMENTS': """
        SELECT doc_id, doc_name, 
               SUBSTRING(content, 1, 500) as content_preview,
               metadata, created_at, updated_at
        FROM documents
        ORDER BY created_at DESC
        LIMIT %s
    """,
    
    'GET_STORAGE_STATISTICS': """
        SELECT 
            (SELECT COUNT(*) FROM documents) as total_documents,
            (SELECT COUNT(*) FROM embeddings) as total_embeddings,
            (SELECT COUNT(*) FROM faiss_indexes) as total_indexes,
            (SELECT SUM(LENGTH(content)) FROM documents) as total_content_size,
            (SELECT SUM(LENGTH(embedding_vector)) FROM embeddings) as total_embedding_size,
            (SELECT SUM(LENGTH(index_data)) FROM faiss_indexes) as total_index_size
    """,
    
    'GET_DOCUMENT_STATISTICS': """
        SELECT 
            COUNT(*) as document_count,
            AVG(LENGTH(content)) as avg_content_length,
            MAX(LENGTH(content)) as max_content_length,
            MIN(LENGTH(content)) as min_content_length
        FROM documents
    """,
    
    'GET_EMBEDDING_STATISTICS': """
        SELECT 
            COUNT(*) as embedding_count,
            AVG(vector_dimension) as avg_dimension,
            MAX(vector_dimension) as max_dimension,
            MIN(vector_dimension) as min_dimension,
            COUNT(DISTINCT doc_id) as unique_documents
        FROM embeddings
    """
}

TABLE_QUERIES = {    
    'ai_subscriptions' : """
        CREATE TABLE `ai_subscriptions` (
        `id` INT(10) UNSIGNED NOT NULL AUTO_INCREMENT,
        `guid` VARCHAR(36) NOT NULL COLLATE 'utf8mb4_general_ci',
        `name` VARCHAR(255) NOT NULL COLLATE 'utf8mb4_general_ci',
        `description` VARCHAR(2000) NULL DEFAULT NULL COLLATE 'utf8mb4_general_ci',
        `account` VARCHAR(30) NOT NULL COLLATE 'utf8mb4_general_ci',
        `api_key` VARCHAR(36) NOT NULL COLLATE 'utf8mb4_general_ci',
        `acl` VARCHAR(255) COLLATE 'utf8mb4_general_ci',
        `model` VARCHAR(64) NULL DEFAULT NULL COLLATE 'utf8mb4_general_ci',
        `length` VARCHAR(16) NULL DEFAULT NULL COLLATE 'utf8mb4_general_ci',
        `reasoning_effort` VARCHAR(16) NULL DEFAULT NULL COLLATE 'utf8mb4_general_ci',
        `expires_at` DATETIME NOT NULL DEFAULT current_timestamp(),
        `created_at` DATETIME NOT NULL DEFAULT current_timestamp(),
        `updated_at` DATETIME NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
        PRIMARY KEY (`id`) USING BTREE,
        UNIQUE KEY `guid` (`guid`),
        UNIQUE KEY `name` (`name`),
        UNIQUE KEY `account` (`account`),
        UNIQUE KEY `api_key` (`api_key`)
    ) COLLATE='utf8mb4_general_ci' ENGINE=InnoDB;
    """,
    
    'openai_prompts': """
        CREATE TABLE `openai_prompts` (
        `id` INT(10) UNSIGNED NOT NULL AUTO_INCREMENT,
        `subscription_id` INT(10) UNSIGNED DEFAULT NULL,
        `guid` VARCHAR(36) NOT NULL COLLATE 'utf8mb4_general_ci',
        `name` VARCHAR(255) NOT NULL COLLATE 'utf8mb4_general_ci',
        `description` VARCHAR(2000) NULL DEFAULT NULL COLLATE 'utf8mb4_general_ci',
        `enabled` tinyint(1) NOT NULL DEFAULT 1,
        `type` VARCHAR(10) NOT NULL,
        `prompt` LONGTEXT NOT NULL COLLATE 'utf8mb4_general_ci',
        `token_count` int(11) DEFAULT NULL,
        `vector` BLOB DEFAULT NULL,
        `question_vector` BLOB DEFAULT NULL,
        `rev` int(11) DEFAULT NULL DEFAULT 1,
        `source` VARCHAR(128) DEFAULT NULL,
        `created_at` DATETIME NOT NULL DEFAULT current_timestamp(),
        `updated_at` DATETIME NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
        PRIMARY KEY (`id`) USING BTREE,
        UNIQUE KEY `prompt_guid` (`subscription_id`,`guid`),
        INDEX `FK_openai_prompts_subscription_id` (`subscription_id`) USING BTREE,
        CONSTRAINT `FK_openai_prompts_subscription_id` FOREIGN KEY (`subscription_id`) REFERENCES `ai_subscriptions` (`id`) ON UPDATE CASCADE ON DELETE CASCADE
        ) COLLATE='utf8mb4_general_ci' ENGINE=InnoDB;
    """,
    
    'openai_debug_logs': """
        CREATE TABLE `openai_debug_logs` (
        `id` INT(10) UNSIGNED NOT NULL AUTO_INCREMENT,
        `subscription_id` INT(10) UNSIGNED DEFAULT NULL,
        `user_guid` VARCHAR(36) NOT NULL COLLATE 'utf8mb4_general_ci',
        `resp_guid` VARCHAR(36) NOT NULL COLLATE 'utf8mb4_general_ci',
        `module` VARCHAR(255) COLLATE 'utf8mb4_general_ci',
        `model` VARCHAR(32) NOT NULL COLLATE 'utf8mb4_general_ci',
        `question` VARCHAR(2000) NULL DEFAULT NULL COLLATE 'utf8mb4_general_ci',
        `messages` LONGTEXT DEFAULT NULL COLLATE 'utf8mb4_general_ci',
        `response` LONGTEXT DEFAULT NULL COLLATE 'utf8mb4_general_ci',
        `created_at` DATETIME NOT NULL DEFAULT current_timestamp(),
        PRIMARY KEY (`id`) USING BTREE,
        INDEX `FK_openai_debug_logs_subscription_id` (`subscription_id`) USING BTREE,
        CONSTRAINT `FK_openai_debug_logs_subscription_id` FOREIGN KEY (`subscription_id`) REFERENCES `ai_subscriptions` (`id`) ON UPDATE CASCADE ON DELETE CASCADE
    ) COLLATE='utf8mb4_general_ci' ENGINE=InnoDB;
    """,
    
    'ai_faiss_indices': """
        CREATE TABLE ai_faiss_indices (
        id               BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        subscription_id  INT(10) UNSIGNED DEFAULT NULL,

        -- 전역 UNIQUE 제거: NOT NULL만 유지
        index_name       VARCHAR(191) NOT NULL,

        index_data       LONGBLOB NOT NULL,                     -- 매우 큰 바이너리 저장
        metadata         JSON NOT NULL,                         -- 일반 메타 (intent/embedding_dim 등)
        artifact_metadata JSON NULL,                            -- 압축/저장소 메타 전용
        rev              INT(11) DEFAULT 1,
        created_at       DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        updated_at       DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),

        -- metadata에서 자주 쓰는 값들을 생성 컬럼으로 노출 (인덱싱 가능)
        total_docs       INT GENERATED ALWAYS AS (
            CAST(JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.total_docs')) AS UNSIGNED)
        ) VIRTUAL,
        embedding_dim    INT GENERATED ALWAYS AS (
            CAST(JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.embedding_dim')) AS UNSIGNED)
        ) VIRTUAL,
        index_type       VARCHAR(64) GENERATED ALWAYS AS (
            JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.index_type'))
        ) VIRTUAL,
        intent           VARCHAR(64) GENERATED ALWAYS AS (
            JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.intent'))
        ) VIRTUAL,
        created_at_meta  DATETIME(6) GENERATED ALWAYS AS (
            STR_TO_DATE(JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.created_at')), '%Y-%m-%dT%H:%i:%s.%f')
        ) VIRTUAL,

        -- 압축/아티팩트 조회용 생성 컬럼 (선택)
        artifact_compressed TINYINT(1)
            GENERATED ALWAYS AS (JSON_EXTRACT(artifact_metadata, '$.compressed') = true) VIRTUAL,
        artifact_codec      VARCHAR(32)
            GENERATED ALWAYS AS (JSON_UNQUOTE(JSON_EXTRACT(artifact_metadata, '$.codec'))) VIRTUAL,
        artifact_size       BIGINT
            GENERATED ALWAYS AS (CAST(JSON_UNQUOTE(JSON_EXTRACT(artifact_metadata, '$.size')) AS UNSIGNED)) VIRTUAL,
        artifact_sha256     VARCHAR(128)
            GENERATED ALWAYS AS (JSON_UNQUOTE(JSON_EXTRACT(artifact_metadata, '$.sha256'))) VIRTUAL,

        -- 구독별 유일성 보장을 위한 가상 컬럼( NULL → -1 치환 )
        subscription_id_key INT GENERATED ALWAYS AS (IFNULL(subscription_id, -1)) VIRTUAL,

        -- 편의 인덱스
        INDEX ix_intent (intent),
        INDEX ix_index_type (index_type),
        INDEX ix_total_docs (total_docs),
        INDEX ix_embedding_dim (embedding_dim),
        INDEX ix_created_at_meta (created_at_meta),

        -- 아티팩트 관련 인덱스(선택)
        INDEX ix_artifact_compressed (artifact_compressed),
        INDEX ix_artifact_codec (artifact_codec),
        INDEX ix_artifact_size (artifact_size),

        -- 구독별 유일성: (subscription_id_key, index_name)
        UNIQUE KEY ux_sub_idx (subscription_id_key, index_name),

        -- 외래 키 설정
        CONSTRAINT fk_ai_faiss_indices_subscription
            FOREIGN KEY (subscription_id)
            REFERENCES ai_subscriptions(id)
            ON DELETE SET NULL
            ON UPDATE CASCADE,

        -- JSON 유효성 체크
        CHECK (JSON_VALID(metadata)),
        CHECK (artifact_metadata IS NULL OR JSON_VALID(artifact_metadata))
        ) ENGINE=InnoDB ROW_FORMAT=DYNAMIC
        DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
    """,
    
    'ai_faiss_index_parts': """
        CREATE TABLE ai_faiss_index_parts (
        id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        subscription_id  INT UNSIGNED    NOT NULL,
        index_name       VARCHAR(191)    NOT NULL,
        part_no          INT UNSIGNED    NOT NULL,        -- 0,1,2,...
        chunk            LONGBLOB        NOT NULL,
        created_at       DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6),

        PRIMARY KEY (id),
        UNIQUE KEY uq_faiss_part (subscription_id, index_name, part_no),
        KEY idx_faiss_lookup (subscription_id, index_name)
        ) ENGINE=InnoDB ROW_FORMAT=DYNAMIC
        DEFAULT CHARSET = utf8mb4
        COLLATE = utf8mb4_unicode_ci;
    """,
    
    'ai_knowledge_graph' : """
        CREATE TABLE ai_knowledge_graph (
        id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        subscription_id INT(10) UNSIGNED DEFAULT NULL,
        data_name       VARCHAR(255) NOT NULL,
        metadata        LONGTEXT NOT NULL,                     -- JSON List 등 대용량 저장 가능
        rev int(11)     DEFAULT 1,
        created_at      DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        updated_at      DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),

        -- 고유 제약: 동일 subscription_id + data_name 중복 방지
        UNIQUE KEY uq_subscription_data_name (subscription_id, data_name),

        -- 외래 키 설정 (ai_subscriptions.id 참조)
        CONSTRAINT fk_ai_knowledge_graph_subscription
            FOREIGN KEY (subscription_id)
            REFERENCES ai_subscriptions(id)
            ON DELETE SET NULL
            ON UPDATE CASCADE,

        -- JSON 유효성 체크
        CHECK (JSON_VALID(metadata))
        ) ENGINE=InnoDB ROW_FORMAT=DYNAMIC
        DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
    """,
    
    'ai_knowledge_graph_parts' : """
        CREATE TABLE ai_knowledge_graph_parts (
        id         BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        kg_id      BIGINT UNSIGNED NOT NULL,              -- ai_knowledge_graph.id
        part_no    INT UNSIGNED NOT NULL,                 -- 0,1,2,...
        codec      ENUM('none','gzip','zstd') DEFAULT 'gzip',
        chunk      LONGBLOB NOT NULL,                     -- 원본 또는 압축/분할 바이트
        checksum   VARBINARY(32) DEFAULT NULL,            -- SHA-256 등 무결성 확인용(옵션)
        created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),

        UNIQUE KEY uq_kg_part (kg_id, part_no),
        CONSTRAINT fk_parts_kg
            FOREIGN KEY (kg_id) REFERENCES ai_knowledge_graph(id)
            ON DELETE CASCADE
            ON UPDATE CASCADE
        ) ENGINE=InnoDB ROW_FORMAT=DYNAMIC
        DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
    """,
}

BATCH_QUERIES = {
    'BATCH_INSERT_EMBEDDINGS': """
        INSERT INTO embeddings 
        (doc_id, chunk_index, chunk_text, embedding_vector, vector_dimension)
        VALUES (%s, %s, %s, %s, %s)
    """,
    
    'BATCH_DELETE_DOCUMENTS': """
        DELETE FROM documents WHERE doc_id IN (%s)
    """,
    
    'BATCH_UPDATE_METADATA': """
        UPDATE documents 
        SET metadata = JSON_SET(metadata, %s, %s)
        WHERE doc_id IN (%s)
    """
}

INDEX_QUERIES = {
    'CREATE_EMBEDDING_VECTOR_INDEX': """
        CREATE INDEX idx_vector_dimension ON embeddings(vector_dimension)
    """,
    
    'CREATE_DOCUMENT_DATE_INDEX': """
        CREATE INDEX idx_created_at ON documents(created_at)
    """,
    
    'CREATE_DOCUMENT_UPDATED_INDEX': """
        CREATE INDEX idx_updated_at ON documents(updated_at)
    """
}
const neo4j = require('neo4j-driver');
const fs = require('fs').promises;

// Neo4j connection configuration
const driver = neo4j.driver(
    'bolt://localhost:7687',
    neo4j.auth.basic('neo4j', 'your_password')
);

// Hàm kiểm tra dữ liệu sản phẩm
function validateProduct(product) {
    if (!product || !product.general) {
        throw new Error('Invalid product data structure');
    }
    if (!product.general.product_id) {
        throw new Error('Missing product_id');
    }
    return true;
}

// Hàm tạo node Product
async function createProductNode(session, product) {
    validateProduct(product);
    
    const query = `
        MERGE (p:Product {id: $id})
        SET p.name = $name,
            p.short_description = $short_description,
            p.product_state = $product_state,
            p.product_condition = $product_condition
        RETURN p
    `;
    
    const params = {
        id: product.general.product_id.toString(), // Convert to string to ensure consistency
        name: product.general.name || '',
        short_description: product.general.attributes?.short_description || '',
        product_state: product.general.attributes?.product_state || '',
        product_condition: product.general.attributes?.product_condition || ''
    };

    console.log('Creating product node with params:', params);
    return await session.run(query, params);
}

// Hàm tạo node Brand
async function createBrandNode(session, product) {
    validateProduct(product);
    
    const query = `
        MERGE (b:Brand {name: $name})
        SET b.type = $type
        RETURN b
    `;
    
    const params = {
        name: product.general.attributes?.manufacturer || 'Unknown',
        type: product.general.attributes?.loaisp || 'Unknown'
    };

    console.log('Creating brand node with params:', params);
    return await session.run(query, params);
}

// Hàm tạo node Category
async function createCategoryNode(session, category) {
    // Kiểm tra và xử lý category data
    if (!category) {
        console.log('Warning: Empty category data');
        return null;
    }

    // Nếu category là string, tạo category đơn giản
    if (typeof category === 'string') {
        const query = `
            MERGE (c:Category {name: $name})
            SET c.id = $id
            RETURN c
        `;
        
        const params = {
            id: category.toLowerCase().replace(/\s+/g, '_'),
            name: category
        };

        console.log('Creating simple category node with params:', params);
        return await session.run(query, params);
    }

    // Nếu category là object nhưng không có categoryId
    if (!category.categoryId) {
        console.log('Warning: Category missing categoryId, using name as ID');
        const query = `
            MERGE (c:Category {name: $name})
            SET c.id = $id,
                c.uri = $uri,
                c.level = $level,
                c.path = $path
            RETURN c
        `;
        
        const params = {
            id: (category.name || 'unknown').toLowerCase().replace(/\s+/g, '_'),
            name: category.name || 'Unknown Category',
            uri: category.uri || '',
            level: category.level || 0,
            path: category.path || ''
        };

        console.log('Creating category node with fallback ID:', params);
        return await session.run(query, params);
    }

    // Xử lý category object đầy đủ
    const query = `
        MERGE (c:Category {id: $id})
        SET c.name = $name,
            c.uri = $uri,
            c.level = $level,
            c.path = $path
        RETURN c
    `;
    
    const params = {
        id: category.categoryId.toString(),
        name: category.name || 'Unknown Category',
        uri: category.uri || '',
        level: category.level || 0,
        path: category.path || ''
    };

    console.log('Creating category node with params:', params);
    return await session.run(query, params);
}

// Hàm tạo node Specification
async function createSpecificationNode(session, product) {
    validateProduct(product);
    
    const specs = [];
    const attrs = product.general.attributes || {};
    
    // Display specs
    if (attrs.display_size) {
        specs.push({
            type: 'display',
            name: 'Display Size',
            value: attrs.display_size,
            unit: 'inches'
        });
    }
    
    // Camera specs
    if (attrs.camera_primary) {
        specs.push({
            type: 'camera',
            name: 'Primary Camera',
            value: attrs.camera_primary
        });
    }
    
    // Performance specs
    if (attrs.chipset) {
        specs.push({
            type: 'performance',
            name: 'Chipset',
            value: attrs.chipset
        });
    }
    
    // Create specification nodes
    for (const spec of specs) {
        const query = `
            MERGE (s:Specification {type: $type, name: $name})
            SET s.value = $value,
                s.unit = $unit
            RETURN s
        `;
        
        console.log('Creating specification node with params:', spec);
        await session.run(query, spec);
    }
    
    return specs;
}

// Hàm tạo node Price
async function createPriceNode(session, product) {
    validateProduct(product);
    
    const query = `
        MERGE (p:Price {product_id: $product_id})
        SET p.base_price = $base_price,
            p.special_price = $special_price,
            p.member_price = $member_price,
            p.vip_price = $vip_price
        RETURN p
    `;
    
    const prices = product.filterable?.prices || {};
    const params = {
        product_id: product.general.product_id.toString(),
        base_price: prices.root?.value || 0,
        special_price: prices.special?.value || 0,
        member_price: prices.smem?.value || 0,
        vip_price: prices.svip?.value || 0
    };

    console.log('Creating price node with params:', params);
    return await session.run(query, params);
}

// Hàm tạo node Review
async function createReviewNode(session, product) {
    validateProduct(product);
    
    const query = `
        MERGE (r:Review {product_id: $product_id})
        SET r.total_count = $total_count,
            r.average_rating = $average_rating
        RETURN r
    `;
    
    const params = {
        product_id: product.general.product_id.toString(),
        total_count: product.general.review?.total_count || 0,
        average_rating: product.general.review?.average_rating || 0
    };

    console.log('Creating review node with params:', params);
    return await session.run(query, params);
}

// Hàm tạo các relationship
async function createRelationships(session, product) {
    validateProduct(product);
    
    const productId = product.general.product_id.toString();
    
    // Product -> Brand
    const brandParams = {
        product_id: productId,
        brand_name: product.general.attributes?.manufacturer || 'Unknown',
        is_official: product.general.attributes?.loaisp === 'Chính hãng',
        warranty_period: product.filterable?.warranty_information || ''
    };
    
    console.log('Creating brand relationship with params:', brandParams);
    await session.run(`
        MATCH (p:Product {id: $product_id})
        MATCH (b:Brand {name: $brand_name})
        MERGE (p)-[r:BELONGS_TO]->(b)
        SET r.is_official = $is_official,
            r.warranty_period = $warranty_period
    `, brandParams);
    
    // Product -> Categories
    if (product.general.categories) {
        for (const category of product.general.categories) {
            try {
                const result = await createCategoryNode(session, category);
                if (result) {
                    const categoryParams = {
                        product_id: productId,
                        category_id: category.categoryId ? category.categoryId.toString() : 
                                   (category.name || 'unknown').toLowerCase().replace(/\s+/g, '_')
                    };
                    console.log('Creating category relationship with params:', categoryParams);
                    await session.run(`
                        MATCH (p:Product {id: $product_id})
                        MATCH (c:Category {id: $category_id})
                        MERGE (p)-[:IN_CATEGORY]->(c)
                    `, categoryParams);
                }
            } catch (error) {
                console.error(`Error processing category for product ${productId}:`, error.message);
                continue;
            }
        }
    }
    
    // Product -> Specifications
    const specs = await createSpecificationNode(session, product);
    for (const spec of specs) {
        const specParams = {
            product_id: productId,
            ...spec
        };
        console.log('Creating specification relationship with params:', specParams);
        await session.run(`
            MATCH (p:Product {id: $product_id})
            MATCH (s:Specification {type: $type, name: $name})
            MERGE (p)-[r:HAS_SPEC]->(s)
            SET r.value = $value
        `, specParams);
    }
    
    // Product -> Price
    const priceParams = { product_id: productId };
    console.log('Creating price relationship with params:', priceParams);
    await session.run(`
        MATCH (p:Product {id: $product_id})
        MATCH (pr:Price {product_id: $product_id})
        MERGE (p)-[:HAS_PRICE]->(pr)
    `, priceParams);
    
    // Product -> Review
    const reviewParams = { product_id: productId };
    console.log('Creating review relationship with params:', reviewParams);
    await session.run(`
        MATCH (p:Product {id: $product_id})
        MATCH (r:Review {product_id: $product_id})
        MERGE (p)-[:HAS_REVIEW]->(r)
    `, reviewParams);
}

// Hàm chính để xây dựng Knowledge Graph
async function buildKnowledgeGraph() {
    const session = driver.session();
    
    try {
        // Đọc dữ liệu từ file
        const data = await fs.readFile('phone_details.json', 'utf8');
        const products = JSON.parse(data);
        
        console.log(`Bắt đầu xây dựng Knowledge Graph cho ${products.length} sản phẩm...`);
        
        for (const product of products) {
            try {
                console.log(`\nĐang xử lý sản phẩm: ${product.general.name}`);
                
                // Tạo các node
                await createProductNode(session, product);
                await createBrandNode(session, product);
                await createPriceNode(session, product);
                await createReviewNode(session, product);
                
                // Tạo các relationship
                await createRelationships(session, product);
                
                console.log(`Đã xử lý xong sản phẩm: ${product.general.name}`);
            } catch (error) {
                console.error(`Lỗi khi xử lý sản phẩm ${product.general.name}:`, error.message);
                continue; // Tiếp tục với sản phẩm tiếp theo nếu có lỗi
            }

            return;
        }
        
        console.log('\nHoàn thành xây dựng Knowledge Graph!');
        
    } catch (error) {
        console.error('Lỗi:', error);
    } finally {
        await session.close();
        await driver.close();
    }
}

// Chạy chương trình
buildKnowledgeGraph();

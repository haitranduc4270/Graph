const neo4j = require('neo4j-driver');

// Neo4j connection configuration
const driver = neo4j.driver(
    'bolt://localhost:7687',
    neo4j.auth.basic('neo4j', 'your_password')
);

// Hàm thực hiện query
async function runQuery(query, params = {}) {
    const session = driver.session();
    try {
        const result = await session.run(query, params);
        return result.records;
    } finally {
        await session.close();
    }
}

// 1. Lấy danh sách tất cả điện thoại
async function getAllPhones() {
    const query = `
        MATCH (p:Product)
        RETURN p.id as id, p.name as name, p.short_description as description
        ORDER BY p.name
    `;
    
    const records = await runQuery(query);
    console.log('\nDanh sách điện thoại:');
    records.forEach(record => {
        console.log(`- ${record.get('name')} (ID: ${record.get('id')})`);
        console.log(`  ${record.get('description')}`);
    });
}

// 2. Tìm điện thoại theo hãng
async function getPhonesByBrand(brandName) {
    const query = `
        MATCH (p:Product)-[:BELONGS_TO]->(b:Brand {name: $brandName})
        RETURN p.id as id, p.name as name, b.name as brand
        ORDER BY p.name
    `;
    
    const records = await runQuery(query, { brandName });
    console.log(`\nĐiện thoại của hãng ${brandName}:`);
    records.forEach(record => {
        console.log(`- ${record.get('name')} (${record.get('brand')})`);
    });
}

// 3. Tìm điện thoại theo khoảng giá
async function getPhonesByPriceRange(minPrice, maxPrice) {
    const query = `
        MATCH (p:Product)-[:HAS_PRICE]->(pr:Price)
        WHERE pr.base_price >= $minPrice AND pr.base_price <= $maxPrice
        RETURN p.id as id, p.name as name, pr.base_price as price
        ORDER BY pr.base_price
    `;
    
    const records = await runQuery(query, { minPrice, maxPrice });
    console.log(`\nĐiện thoại trong khoảng giá ${minPrice} - ${maxPrice}:`);
    records.forEach(record => {
        console.log(`- ${record.get('name')}: ${record.get('price').toLocaleString()}đ`);
    });
}

// 4. Tìm điện thoại theo thông số kỹ thuật
async function getPhonesBySpec(specType, specValue) {
    const query = `
        MATCH (p:Product)-[r:HAS_SPEC]->(s:Specification {type: $specType})
        WHERE s.value CONTAINS $specValue
        RETURN p.id as id, p.name as name, s.name as spec_name, s.value as spec_value
        ORDER BY p.name
    `;
    
    const records = await runQuery(query, { specType, specValue });
    console.log(`\nĐiện thoại có ${specType} chứa "${specValue}":`);
    records.forEach(record => {
        console.log(`- ${record.get('name')}: ${record.get('spec_name')} = ${record.get('spec_value')}`);
    });
}

// 5. Tìm điện thoại theo đánh giá
async function getPhonesByRating(minRating) {
    const query = `
        MATCH (p:Product)-[:HAS_REVIEW]->(r:Review)
        WHERE r.average_rating >= $minRating
        RETURN p.id as id, p.name as name, r.average_rating as rating, r.total_count as review_count
        ORDER BY r.average_rating DESC
    `;
    
    const records = await runQuery(query, { minRating });
    console.log(`\nĐiện thoại có đánh giá từ ${minRating} sao trở lên:`);
    records.forEach(record => {
        console.log(`- ${record.get('name')}: ${record.get('rating')} sao (${record.get('review_count')} đánh giá)`);
    });
}

// 6. Tìm điện thoại theo danh mục
async function getPhonesByCategory(categoryName) {
    const query = `
        MATCH (p:Product)-[:IN_CATEGORY]->(c:Category)
        WHERE c.name CONTAINS $categoryName
        RETURN p.id as id, p.name as name, c.name as category
        ORDER BY p.name
    `;
    
    const records = await runQuery(query, { categoryName });
    console.log(`\nĐiện thoại trong danh mục "${categoryName}":`);
    records.forEach(record => {
        console.log(`- ${record.get('name')} (${record.get('category')})`);
    });
}

// 7. Tìm điện thoại chính hãng của một thương hiệu
async function getOfficialPhonesByBrand(brandName) {
    const query = `
        MATCH (p:Product)-[r:BELONGS_TO]->(b:Brand {name: $brandName})
        WHERE r.is_official = true
        RETURN p.id as id, p.name as name, r.warranty_period as warranty
        ORDER BY p.name
    `;
    
    const records = await runQuery(query, { brandName });
    console.log(`\nĐiện thoại chính hãng của ${brandName}:`);
    records.forEach(record => {
        console.log(`- ${record.get('name')} (Bảo hành: ${record.get('warranty')})`);
    });
}

// Chạy các ví dụ
async function runExamples() {
    try {
        // Ví dụ 1: Lấy tất cả điện thoại
        await getAllPhones();
        
        // Ví dụ 2: Tìm điện thoại Samsung
        await getPhonesByBrand('Samsung');
        
        // Ví dụ 3: Tìm điện thoại trong khoảng giá 5-10 triệu
        await getPhonesByPriceRange(5000000, 10000000);
        
        // Ví dụ 4: Tìm điện thoại có màn hình 6.7 inch
        await getPhonesBySpec('display', '6.7');
        
        // Ví dụ 5: Tìm điện thoại có đánh giá từ 4.5 sao
        await getPhonesByRating(4.5);
        
        // Ví dụ 6: Tìm điện thoại trong danh mục "iPhone"
        await getPhonesByCategory('iPhone');
        
        // Ví dụ 7: Tìm điện thoại chính hãng của Apple
        await getOfficialPhonesByBrand('Apple');
        
    } catch (error) {
        console.error('Lỗi:', error);
    } finally {
        await driver.close();
    }
}

// Chạy chương trình
runExamples(); 
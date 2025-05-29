const axios = require("axios");
const fs = require("fs");
const path = require("path");

const filePath = path.join(__dirname, "phones_all.json");
const outputPath = path.join(__dirname, "phone_details.json");

// Hàm delay
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Hàm lấy chi tiết sản phẩm theo id
const getProductDetail = async (productId) => {
  const query = `
    query getProductDataDetail {
      product(
        id: ${productId},
        provinceId: 30
      ) {
        general {
          name
          attributes
          product_id
          categories {
            name
            uri
          }
          review {
            total_count
            average_rating
          }
        }
        filterable {
          promotion_pack
          price
          prices
          warranty_information
        }
      }
    }
  `;

  try {
    const response = await axios.post(
      "https://api.cellphones.com.vn/v2/graphql/query",
      { query, variables: {} },
      {
        headers: {
          accept: "application/json",
          "content-type": "application/json",
          origin: "https://cellphones.com.vn",
          referer: "https://cellphones.com.vn/",
          "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        },
      }
    );
    return response.data.data.product;
  } catch (error) {
    console.error(`Lỗi lấy chi tiết productId=${productId}: ${error.message}`);
    return null;
  }
};

// Hàm xử lý một chunk sản phẩm
const processChunk = async (products, startIndex, chunkSize) => {
  const chunk = products.slice(startIndex, startIndex + chunkSize);
  const results = [];
  
  for (const p of chunk) {
    const productId = p.general.product_id;
    console.log(`Lấy chi tiết sản phẩm id: ${productId}`);
    
    const detail = await getProductDetail(productId);
    if (detail) {
      results.push(detail);
    }
  }
  
  return results;
};

const main = async () => {
  if (!fs.existsSync(filePath)) {
    console.error("File phones_all.json không tồn tại.");
    return;
  }

  const rawData = fs.readFileSync(filePath, "utf-8");
  const products = JSON.parse(rawData);
  
  const chunkSize = 50;
  const totalChunks = Math.ceil(products.length / chunkSize);
  let allResults = [];
  
  for (let i = 0; i < totalChunks; i++) {
    console.log(`\nXử lý chunk ${i + 1}/${totalChunks}`);
    const startIndex = i * chunkSize;
    
    // Xử lý chunk hiện tại
    const chunkResults = await processChunk(products, startIndex, chunkSize);
    allResults = allResults.concat(chunkResults);
    
    // Lưu kết quả tạm thời sau mỗi chunk
    fs.writeFileSync(outputPath, JSON.stringify(allResults, null, 2), "utf-8");
    console.log(`Đã lưu ${allResults.length} sản phẩm vào file.`);
    
    // Delay 10s trước khi xử lý chunk tiếp theo (nếu không phải chunk cuối)
    if (i < totalChunks - 1) {
      console.log("Đợi 10 giây trước khi xử lý chunk tiếp theo...");
      await delay(10000);
    }
  }

  console.log("\nHoàn tất lấy chi tiết sản phẩm!");
  console.log(`Tổng số sản phẩm đã lưu: ${allResults.length}`);
};

main();
